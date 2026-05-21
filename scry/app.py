"""FastAPI surface for Scry.

NOT IMPLEMENTED — foundation only. The interface below is the contract the
runners and frontend (eventually) will target. Implement piecewise per the
ROADMAP in README.md.

Mirrors Merlina's app.py shape so a contributor can borrow patterns directly:
- Job queue (single-GPU at a time, sqlite-backed history)
- WebSocket progress updates
- POST /eval -> 202 + job_id
- GET /jobs / GET /jobs/{id} / POST /jobs/{id}/stop
- GET /results/{id} (zipped JSON + markdown)
"""
from __future__ import annotations

# Intentional stub. See README "Roadmap" for milestones.

if __name__ == "__main__":
    raise SystemExit(
        "scry.app is a foundation-only stub. Nothing to run yet.\n"
        "See README.md for the planned API surface and roadmap."
    )
