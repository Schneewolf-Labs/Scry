"""Tests for the v0.2 FastAPI surface (requires the `api` extra + httpx)."""
import threading

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import FakeRunner, make_config, wait_for
from scry.app import create_app
from scry.jobs import JobManager


@pytest.fixture
def make_client(tmp_path):
    created = []

    def _make(runner=None):
        mgr = JobManager(
            db_path=tmp_path / f"jobs{len(created)}.db",
            runners=[runner or FakeRunner()],
        )
        client = TestClient(create_app(manager=mgr))
        created.append((client, mgr))
        return client

    yield _make
    for client, mgr in created:
        client.close()
        mgr.shutdown()


def submit(client, *tasks, **cfg_kwargs) -> str:
    response = client.post("/eval", json=make_config(*tasks, **cfg_kwargs).model_dump())
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    return body["job_id"]


def wait_status(client, job_id, *statuses, timeout=10.0):
    return wait_for(
        lambda: (job := client.get(f"/jobs/{job_id}").json())["status"] in statuses
        and job,
        timeout=timeout,
    )


def test_health(make_client):
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_submit_then_fetch_results(make_client):
    client = make_client(FakeRunner(score=0.9))
    job_id = submit(client, "task_a", "task_b")

    job = wait_status(client, job_id, "completed")
    assert job["results"]["summary"] == {"task_a": 0.9, "task_b": 0.9}

    results = client.get(f"/results/{job_id}")
    assert results.status_code == 200
    assert results.json()["results"]["summary"] == {"task_a": 0.9, "task_b": 0.9}


def test_results_conflict_while_in_flight(make_client):
    gate = threading.Event()
    client = make_client(FakeRunner(gate=gate))
    job_id = submit(client)

    assert client.get(f"/results/{job_id}").status_code == 409
    gate.set()
    wait_status(client, job_id, "completed")
    assert client.get(f"/results/{job_id}").status_code == 200


def test_list_jobs_newest_first(make_client):
    client = make_client()
    first = submit(client, "task_a")
    second = submit(client, "task_b")
    wait_status(client, second, "completed")

    jobs = client.get("/jobs").json()["jobs"]
    assert [job["id"] for job in jobs[:2]] == [second, first]


def test_unknown_job_is_404(make_client):
    client = make_client()
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/results/nope").status_code == 404
    assert client.post("/jobs/nope/stop").status_code == 404


def test_invalid_config_is_422(make_client):
    client = make_client()
    response = client.post("/eval", json={"base_model": "m"})  # no benchmarks
    assert response.status_code == 422


def test_stop_queued_job(make_client):
    gate = threading.Event()
    client = make_client(FakeRunner(gate=gate))
    submit(client, "blocker")
    victim = submit(client, "victim")

    response = client.post(f"/jobs/{victim}/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    gate.set()


def test_hf_token_never_leaves_the_api(make_client):
    client = make_client()
    job_id = submit(client, "task_a", hf_token="hf_secret123")
    wait_status(client, job_id, "completed")

    for path in (f"/jobs/{job_id}", "/jobs", f"/results/{job_id}"):
        assert "hf_secret123" not in client.get(path).text


def test_websocket_streams_progress_to_terminal(make_client):
    gate = threading.Event()
    client = make_client(FakeRunner(gate=gate))
    job_id = submit(client)

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        gate.set()

        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event.get("type") in ("status", "snapshot") and (
                event.get("status") in ("completed", "failed", "cancelled")
                or event.get("job", {}).get("status")
                in ("completed", "failed", "cancelled")
            ):
                break

    types = {event["type"] for event in events}
    assert "progress" in types or "benchmark_completed" in types or "snapshot" in types
    last = events[-1]
    final_status = last.get("status") or last["job"]["status"]
    assert final_status == "completed"


def test_websocket_on_finished_job_sends_snapshot_and_closes(make_client):
    client = make_client()
    job_id = submit(client)
    wait_status(client, job_id, "completed")

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert snapshot["job"]["status"] == "completed"
