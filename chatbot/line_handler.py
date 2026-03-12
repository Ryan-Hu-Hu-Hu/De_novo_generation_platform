"""LINE conversation state machine and message helpers."""

import os
import re
import logging
from typing import Optional

from linebot.v3.messaging import (
    ApiClient, Configuration, MessagingApi,
    ReplyMessageRequest, PushMessageRequest,
    TextMessage, QuickReply, QuickReplyItem,
    MessageAction,
)

from pipeline.pdb_utils import validate_pdb_id
from . import job_manager

logger = logging.getLogger(__name__)

# Temperature options for quick reply
TEMP_OPTIONS = list(range(10, 110, 10))  # [10, 20, ..., 100]


def _make_api_client() -> MessagingApi:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    config = Configuration(access_token=token)
    return MessagingApi(ApiClient(config))


def _temp_quick_reply() -> QuickReply:
    items = [
        QuickReplyItem(
            action=MessageAction(label=f"{t}°C", text=str(t))
        )
        for t in TEMP_OPTIONS
    ]
    return QuickReply(items=items)


def reply(reply_token: str, text: str, quick_reply: Optional[QuickReply] = None) -> None:
    """Send a reply message to a LINE user."""
    api = _make_api_client()
    msg = TextMessage(text=text, quick_reply=quick_reply)
    req = ReplyMessageRequest(reply_token=reply_token, messages=[msg])
    try:
        api.reply_message(req)
    except Exception as exc:
        logger.error("Failed to send reply: %s", exc)


def push_message(user_id: str, text: str, quick_reply: Optional[QuickReply] = None) -> None:
    """Push a message to a LINE user (used for async job updates)."""
    api = _make_api_client()
    msg = TextMessage(text=text, quick_reply=quick_reply)
    req = PushMessageRequest(to=user_id, messages=[msg])
    try:
        api.push_message(req)
    except Exception as exc:
        logger.error("Failed to push message to %s: %s", user_id, exc)


# ── State machine ─────────────────────────────────────────────────────────────

def handle_message(user_id: str, reply_token: str, text: str) -> None:
    """
    Route incoming text to the appropriate state handler.

    States:
        IDLE             → prompt for PDB code
        awaiting_pdb     → validate PDB, move to awaiting_temperature
        awaiting_temp    → validate temp, submit job, move to processing
        processing       → show current status
    """
    user = job_manager.get_user_state(user_id)
    state = user.get("state", "IDLE")
    text = text.strip()

    # Global commands available from any state
    if text.lower() in ("help", "?", "/help"):
        reply(reply_token, _help_text())
        return

    if text.lower() in ("cancel", "reset", "stop"):
        job_manager.set_user_state(user_id, "IDLE", pdb_id=None, active_job=None)
        reply(reply_token, "Cancelled. Send any message to start a new job.")
        return

    if state == "IDLE":
        _handle_idle(user_id, reply_token, text)
    elif state == "awaiting_pdb":
        _handle_awaiting_pdb(user_id, reply_token, text)
    elif state == "awaiting_temp":
        _handle_awaiting_temp(user_id, reply_token, text)
    elif state == "processing":
        _handle_processing(user_id, reply_token, text)
    else:
        # Unknown state — reset
        job_manager.set_user_state(user_id, "IDLE")
        reply(reply_token, "Welcome! Please enter a 4-letter PDB code to begin.")


def _help_text() -> str:
    return (
        "De Novo Protein Generator — Commands\n"
        "─────────────────────────────────\n"
        "start / (any message)\n"
        "  → Begin a new generation job\n\n"
        "cancel / reset / stop\n"
        "  → Cancel current job and return to start\n\n"
        "help / ?\n"
        "  → Show this message\n\n"
        "─────────────────────────────────\n"
        "Workflow:\n"
        "1. Enter a 4-letter PDB code (e.g. 1PMO)\n"
        "2. Select a reaction temperature (10–100 °C)\n"
        "3. Wait for the pipeline to run\n"
        "4. Receive the best candidate sequence\n\n"
        "The pipeline may take several minutes.\n"
        "Progress updates will be sent automatically."
    )


def _handle_idle(user_id: str, reply_token: str, text: str) -> None:
    # Any non-empty message starts the conversation
    job_manager.set_user_state(user_id, "awaiting_pdb")
    reply(
        reply_token,
        "Welcome to the De Novo Protein Generator!\n\n"
        "Please enter the 4-letter PDB code of your template protein "
        "(e.g. 1PMO).\n\n"
        "Type 'help' at any time to see available commands."
    )


def _handle_awaiting_pdb(user_id: str, reply_token: str, text: str) -> None:
    pdb_id = text.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{4}", pdb_id):
        reply(reply_token, "That doesn't look like a valid PDB code. Please enter a 4-letter code (e.g. 1PMO):")
        return

    # Consume the reply token with the "validating" message, then push the result
    reply(reply_token, f"Validating {pdb_id} with RCSB…")
    if not validate_pdb_id(pdb_id):
        push_message(
            user_id,
            f"PDB entry {pdb_id} was not found in RCSB. "
            "Please double-check and try again:"
        )
        return

    job_manager.set_user_state(user_id, "awaiting_temp", pdb_id=pdb_id)
    push_message(
        user_id,
        f"✓ Found {pdb_id}.\n\n"
        "What is the desired reaction temperature (°C)?\n"
        "Select from the quick replies or type a number between 10 and 100:",
        quick_reply=_temp_quick_reply(),
    )


def _handle_awaiting_temp(user_id: str, reply_token: str, text: str) -> None:
    try:
        temp = int(text)
        if not (10 <= temp <= 100):
            raise ValueError
    except ValueError:
        reply(
            reply_token,
            "Please enter a temperature between 10 and 100 °C:",
            quick_reply=_temp_quick_reply(),
        )
        return

    user = job_manager.get_user_state(user_id)
    pdb_id = user.get("pdb_id")
    if not pdb_id:
        job_manager.set_user_state(user_id, "IDLE")
        reply(reply_token, "Session expired. Please start again by sending any message.")
        return

    reply(
        reply_token,
        f"Starting de novo generation for {pdb_id} at {temp}°C.\n"
        "This may take a while. I will keep you updated on progress."
    )

    job_manager.submit_job(
        user_id=user_id,
        pdb_id=pdb_id,
        target_temp=temp,
        push_message=push_message,
    )


def _handle_processing(user_id: str, reply_token: str, text: str) -> None:
    job = job_manager.get_user_active_job(user_id)
    if not job:
        job_manager.set_user_state(user_id, "IDLE")
        reply(reply_token, "No active job found. Send me a PDB code to start.")
        return

    status = job.get("status", "unknown")
    if status == "done":
        import json
        result = json.loads(job.get("result") or "{}")
        if result.get("sequence"):
            msg = (
                f"Your job is complete!\n"
                f"Best candidate: {result['name']}\n"
                f"Sequence: {result['sequence'][:80]}...\n"
                f"EC: {result.get('ec', 'N/A')}\n"
                f"Topt: {result.get('topt', 'N/A')}°C"
            )
        else:
            msg = result.get("message", "No candidates found.")
        job_manager.set_user_state(user_id, "IDLE")
        reply(reply_token, msg)
    elif status == "failed":
        job_manager.set_user_state(user_id, "IDLE")
        reply(reply_token, f"Your job failed: {job.get('error', 'unknown error')}. Please try again.")
    else:
        reply(
            reply_token,
            f"Your job is still running (status: {status}). Please wait…\n\n"
            "Type 'cancel' to stop and start a new job.",
        )
