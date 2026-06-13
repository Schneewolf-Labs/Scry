"""Console entrypoint for the Scry API server.

v0.2: serves the FastAPI app (job queue + WebSocket progress) via uvicorn.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scry API server - job queue + WebSocket eval progress"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--db",
        default="data/eval_jobs.db",
        help="Path to the sqlite job-history database (default: data/eval_jobs.db)"
    )
    args = parser.parse_args()

    try:
        import uvicorn

        from .app import create_app
    except ImportError:
        raise SystemExit(
            "The Scry API server requires the 'api' extra.\n"
            "Install with: pip install 'scry[api]'"
        )

    uvicorn.run(create_app(db_path=args.db), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
