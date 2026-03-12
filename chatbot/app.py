"""Flask webhook server for the LINE Messaging API."""

import os
import logging
from flask import Flask, request, abort

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from . import line_handler
from chatbot import job_manager

logger = logging.getLogger(__name__)

app = Flask(__name__)

_handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET", ""))


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/callback", methods=["POST"])
def callback():
    """Receive LINE webhook events and dispatch to the handler."""
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    logger.debug("Webhook body: %s", body[:200])

    try:
        _handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid LINE signature")
        abort(400)

    return "OK", 200


@_handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_id    = event.source.user_id
    reply_token = event.reply_token
    text       = event.message.text or ""
    line_handler.handle_message(user_id, reply_token, text)
