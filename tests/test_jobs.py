"""Tests for the v0.2 job queue: lifecycle, cancellation, events, persistence."""
import threading

import pytest

from conftest import FakeRunner, make_config, wait_for
from scry.config import BenchmarkSpec, EvalConfig
from scry.jobs import TERMINAL_STATUSES, JobManager, JobStatus, JobStore


@pytest.fixture
def make_manager(tmp_path):
    """Factory so each test picks its runner; managers are shut down after."""
    managers = []

    def _make(runner=None, db_name=None):
        db_path = tmp_path / (db_name or f"jobs{len(managers)}.db")
        mgr = JobManager(db_path=db_path, runners=[runner or FakeRunner()])
        managers.append(mgr)
        return mgr

    yield _make
    for mgr in managers:
        mgr.shutdown()


def wait_terminal(mgr, job_id, timeout=10.0):
    return wait_for(
        lambda: (job := mgr.store.get(job_id))
        and job.status in TERMINAL_STATUSES and job,
        timeout=timeout,
    )


class TestJobLifecycle:
    def test_job_completes_with_results(self, make_manager):
        mgr = make_manager(FakeRunner(score=0.75))
        job = mgr.submit(make_config("task_a"))
        assert job.status is JobStatus.QUEUED

        done = wait_terminal(mgr, job.id)
        assert done.status is JobStatus.COMPLETED
        assert done.results["summary"] == {"task_a": 0.75}
        assert done.results["base_model"] == "schneewolflabs/test-model"
        assert len(done.results["benchmarks"]) == 1
        assert done.started_at is not None
        assert done.finished_at is not None
        assert done.error is None

    def test_multi_benchmark_runs_all_specs_in_order(self, make_manager):
        runner = FakeRunner()
        mgr = make_manager(runner)
        job = mgr.submit(make_config("task_a", "task_b", "task_c"))

        done = wait_terminal(mgr, job.id)
        assert done.status is JobStatus.COMPLETED
        assert runner.calls == ["task_a", "task_b", "task_c"]
        assert set(done.results["summary"]) == {"task_a", "task_b", "task_c"}

    def test_failing_benchmark_marks_job_failed_keeps_partials(self, make_manager):
        mgr = make_manager(FakeRunner(fail_tasks=("task_b",)))
        job = mgr.submit(make_config("task_a", "task_b", "task_c"))

        done = wait_terminal(mgr, job.id)
        assert done.status is JobStatus.FAILED
        assert "task_b" in done.error and "boom" in done.error
        # task_a finished before the failure; task_c never ran.
        assert done.results["summary"] == {"task_a": 0.5}

    def test_unknown_backend_fails_job(self, make_manager):
        mgr = make_manager(FakeRunner())
        cfg = EvalConfig(
            base_model="schneewolflabs/test-model",
            benchmarks=[BenchmarkSpec(backend="bfcl", task="single_function_call")],
        )
        done = wait_terminal(mgr, mgr.submit(cfg).id)
        assert done.status is JobStatus.FAILED
        assert "No runner available" in done.error

    def test_jobs_run_fifo(self, make_manager):
        runner = FakeRunner()
        mgr = make_manager(runner)
        first = mgr.submit(make_config("first"))
        second = mgr.submit(make_config("second"))
        wait_terminal(mgr, first.id)
        wait_terminal(mgr, second.id)
        assert runner.calls == ["first", "second"]


class TestStop:
    def test_stop_queued_job_cancels_without_running(self, make_manager):
        gate = threading.Event()
        runner = FakeRunner(gate=gate)
        mgr = make_manager(runner)
        blocker = mgr.submit(make_config("blocker"))
        victim = mgr.submit(make_config("victim"))

        stopped = mgr.stop(victim.id)
        assert stopped.status is JobStatus.CANCELLED

        gate.set()
        wait_terminal(mgr, blocker.id)
        assert "victim" not in runner.calls
        assert mgr.store.get(victim.id).status is JobStatus.CANCELLED

    def test_stop_running_job_halts_after_current_benchmark(self, make_manager):
        gate, started = threading.Event(), threading.Event()
        runner = FakeRunner(gate=gate, started=started)
        mgr = make_manager(runner)
        job = mgr.submit(make_config("task_a", "task_b"))

        assert started.wait(timeout=10)
        stopped = mgr.stop(job.id)
        assert stopped.status is JobStatus.RUNNING  # halts at the boundary
        gate.set()

        done = wait_terminal(mgr, job.id)
        assert done.status is JobStatus.CANCELLED
        assert runner.calls == ["task_a"]  # task_b skipped
        assert done.results["summary"] == {"task_a": 0.5}

    def test_stop_finished_job_is_noop(self, make_manager):
        mgr = make_manager()
        job = mgr.submit(make_config())
        wait_terminal(mgr, job.id)
        assert mgr.stop(job.id).status is JobStatus.COMPLETED

    def test_stop_unknown_job_raises(self, make_manager):
        with pytest.raises(KeyError):
            make_manager().stop("nope")


class TestEvents:
    def test_subscribers_see_full_event_stream(self, make_manager):
        gate = threading.Event()
        mgr = make_manager(FakeRunner(gate=gate))
        job = mgr.submit(make_config("task_a"))
        events_q = mgr.subscribe(job.id)
        gate.set()

        events = []
        while True:
            event = events_q.get(timeout=10)
            events.append(event)
            if (event["type"] == "status"
                    and event["status"] in {s.value for s in TERMINAL_STATUSES}):
                break

        types = [e["type"] for e in events]
        assert "benchmark_started" in types
        assert "progress" in types
        assert "benchmark_completed" in types
        assert events[-1]["status"] == "completed"
        assert events[-1]["summary"] == {"task_a": 0.5}
        mgr.unsubscribe(job.id, events_q)


class TestPersistence:
    def test_results_survive_in_new_store(self, make_manager, tmp_path):
        mgr = make_manager(db_name="shared.db")
        job = mgr.submit(make_config("task_a"))
        wait_terminal(mgr, job.id)

        reread = JobStore(tmp_path / "shared.db").get(job.id)
        assert reread.status is JobStatus.COMPLETED
        assert reread.results["summary"] == {"task_a": 0.5}

    def test_recovery_on_restart(self, tmp_path):
        """Jobs left 'running' fail on boot; queued jobs are re-enqueued."""
        db_path = tmp_path / "recover.db"
        store = JobStore(db_path)
        interrupted = store.create(make_config("interrupted"))
        store.update(interrupted.id, status=JobStatus.RUNNING)
        pending = store.create(make_config("pending"))
        store.close()

        mgr = JobManager(db_path=db_path, runners=[FakeRunner()])
        try:
            failed = mgr.store.get(interrupted.id)
            assert failed.status is JobStatus.FAILED
            assert "restart" in failed.error

            done = wait_terminal(mgr, pending.id)
            assert done.status is JobStatus.COMPLETED
        finally:
            mgr.shutdown()


class TestJobShape:
    def test_to_dict_masks_hf_token(self, make_manager):
        mgr = make_manager()
        job = mgr.submit(make_config("task_a", hf_token="hf_secret123"))
        wait_terminal(mgr, job.id)

        public = mgr.store.get(job.id).to_dict()
        assert public["config"]["hf_token"] == "***"
        assert "hf_secret123" not in str(public)
