"""SQLite-backed job manager with thread-pool execution."""

import os
import uuid
import sqlite3
import logging
import threading
from typing import Optional, Callable

from pipeline.config import JOBS_DB_PATH, DATA_DIR
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    pdb_id      TEXT,
    target_temp INTEGER,
    result      TEXT,
    error       TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS user_state (
    user_id     TEXT PRIMARY KEY,
    state       TEXT NOT NULL DEFAULT 'IDLE',
    pdb_id      TEXT,
    active_job  TEXT
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(JOBS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_STATE_TABLE)
        conn.commit()


# ── User conversation state ───────────────────────────────────────────────────

def get_user_state(user_id: str) -> dict:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM user_state WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row:
        return dict(row)
    return {"user_id": user_id, "state": "IDLE", "pdb_id": None, "active_job": None}


def set_user_state(user_id: str, state: str, pdb_id: Optional[str] = None, active_job: Optional[str] = None) -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_state (user_id, state, pdb_id, active_job)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                state = excluded.state,
                pdb_id = COALESCE(excluded.pdb_id, pdb_id),
                active_job = excluded.active_job
            """,
            (user_id, state, pdb_id, active_job),
        )
        conn.commit()


# ── Job CRUD ──────────────────────────────────────────────────────────────────

def create_job(user_id: str, pdb_id: str, target_temp: int) -> str:
    job_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, user_id, status, pdb_id, target_temp) VALUES (?, ?, 'pending', ?, ?)",
            (job_id, user_id, pdb_id, target_temp),
        )
        conn.commit()
    return job_id


def update_job(job_id: str, **kwargs) -> None:
    allowed = {"status", "result", "error"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    placeholders = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [job_id]
    with _get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {placeholders} WHERE job_id = ?", values)
        conn.commit()


def get_job(job_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


JOB_TIMEOUT_HOURS = 2  # Jobs running longer than this are considered stale


def expire_stale_jobs(user_id: str) -> None:
    """Mark jobs that have been running/pending for too long as failed."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE jobs SET status = 'failed', error = 'Job timed out (stale from previous session)'
            WHERE user_id = ? AND status NOT IN ('done', 'failed')
              AND created_at < datetime('now', ? || ' hours')
            """,
            (user_id, f"-{JOB_TIMEOUT_HOURS}"),
        )
        conn.commit()


def get_user_active_job(user_id: str) -> Optional[dict]:
    expire_stale_jobs(user_id)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE user_id = ? AND status NOT IN ('done', 'failed') "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


# ── Job execution ─────────────────────────────────────────────────────────────

def submit_job(
    user_id: str,
    pdb_id: str,
    target_temp: int,
    push_message: Callable[[str, str], None],
) -> str:
    """
    Create a job record, spawn a background thread, and return job_id.

    *push_message(user_id, text)* is called for progress updates.
    """
    job_id = create_job(user_id, pdb_id, target_temp)
    set_user_state(user_id, "processing", active_job=job_id)

    def _run():
        update_job(job_id, status="running")
        job_dir = os.path.join(DATA_DIR, "jobs", job_id)
        orchestrator = PipelineOrchestrator()

        def progress(msg: str):
            logger.info("[job %s] %s", job_id, msg)
            push_message(user_id, msg)

        try:
            result = orchestrator.run(pdb_id, target_temp, job_dir, progress)
            import json
            update_job(job_id, status="done", result=json.dumps(result))
            set_user_state(user_id, "IDLE", active_job=None)

            # Format and push final result
            if result.get("sequence"):
                msg = (
                    f"Job complete!\n"
                    f"Sequence: {result['sequence'][:60]}...\n"
                    f"EC: {result.get('ec', 'N/A')}\n"
                    f"Solubility: {result.get('solubility', 'N/A'):.3f}\n"
                    f"Topt: {result.get('topt', 'N/A'):.1f}°C\n"
                    f"kcat: {result.get('kcat', 'N/A')}\n"
                    f"Km: {result.get('Km', 'N/A')}"
                )
            else:
                msg = result.get("message", "No candidates found.")
            push_message(user_id, msg)

        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            update_job(job_id, status="failed", error=str(exc))
            set_user_state(user_id, "IDLE", active_job=None)
            push_message(user_id, f"Sorry, the pipeline encountered an error: {exc}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job_id
