"""Project past meme operations without changing archived conversation records."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from contextvars import ContextVar

TRANSIENT_TOOLS = {"meme_search", "meme_view", "meme_send"}
SENT_HEADER = "[历史记录：机器人已发送图片；以下是历史附件，不是用户新发言]"
CACHED_PATH = re.compile(r"Image returned and cached at path='([^']+)'")
IMAGE_HEADER = re.compile(r"\[Image from tool 'meme_view', path='([^']+)'\]")


def as_dict(message):
    return (
        message.model_dump()
        if hasattr(message, "model_dump")
        else copy.deepcopy(message)
    )


def text_content(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def tool_data(message):
    try:
        data, _ = json.JSONDecoder().raw_decode(text_content(message).lstrip())
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def tool_results(messages):
    names = {}
    for raw in messages:
        message = as_dict(raw)
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                names[call.get("id")] = call.get("function", {}).get("name")
        elif message.get("role") == "tool":
            yield names.pop(message.get("tool_call_id"), None), message


def result_key(message):
    return json.dumps(message, ensure_ascii=False, sort_keys=True)


def view_image_path(parts, index):
    part = parts[index]
    header = (
        IMAGE_HEADER.fullmatch(part.get("text", ""))
        if part.get("type") == "text"
        else None
    )
    following = parts[index + 1] if index + 1 < len(parts) else {}
    image = following.get("image_url", {})
    if (
        header
        and following.get("type") == "image_url"
        and isinstance(image, dict)
        and image.get("id") == header[1]
    ):
        return header[1]
    return None


class HistoryProjection:
    """Build visible context messages and keep each original record's index."""

    def __init__(self, messages, receipts):
        self.receipts = receipts
        self.paths = set()
        self.active = any(
            call.get("function", {}).get("name") in TRANSIENT_TOOLS
            for raw in messages
            for call in as_dict(raw).get("tool_calls") or []
        )
        for name, message in tool_results(messages):
            self.active |= name in TRANSIENT_TOOLS
            if name == "meme_view":
                self.paths.update(CACHED_PATH.findall(text_content(message)))
        # A prior truncation may have removed the tool pair but retained its
        # image envelope. Require both the native header and matching image ID.
        for raw in messages:
            message = as_dict(raw)
            if message.get("role") == "user" and isinstance(
                message.get("content"), list
            ):
                parts = message["content"]
                for index in range(len(parts)):
                    path = view_image_path(parts, index)
                    if path is not None:
                        self.paths.add(path)
                        self.active = True

    def project(self, messages):
        return [message for _, message in self.entries(messages)]

    def entries(self, messages):
        projected = []
        names = {}
        for source, raw in enumerate(messages):
            message = as_dict(raw)
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                kept = []
                removed = False
                for call in message["tool_calls"]:
                    name = call.get("function", {}).get("name")
                    names[call.get("id")] = name
                    if name in TRANSIENT_TOOLS:
                        removed = True
                    else:
                        kept.append(call)
                if removed:
                    if kept:
                        message["tool_calls"] = kept
                    else:
                        message.pop("tool_calls", None)
                        content = message.get("content")
                        if isinstance(content, list):
                            message["content"] = [
                                p for p in content if p.get("type") != "think"
                            ]
                        if not message.get("content"):
                            continue
            elif role == "tool":
                name = names.pop(message.get("tool_call_id"), None)
                if name in TRANSIENT_TOOLS:
                    receipt = self.receipts.get(result_key(message))
                    if name == "meme_send" and receipt is not None:
                        projected.append((source, copy.deepcopy(receipt)))
                    continue
            elif role == "user" and isinstance(message.get("content"), list):
                parts = message["content"]
                kept = []
                index = 0
                while index < len(parts):
                    part = parts[index]
                    path = view_image_path(parts, index)
                    if path is not None and path in self.paths:
                        index += 2
                        continue
                    kept.append(part)
                    index += 1
                if not kept:
                    continue
                message["content"] = kept
            projected.append((source, message))
        return projected


def sent_message(data, frames):
    metadata = {
        key: data[key]
        for key in ("id", "name", "description", "sent_at")
        if key in data
    }
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": SENT_HEADER + "\n" + json.dumps(metadata, ensure_ascii=False),
            },
            *[{"type": "image_url", "image_url": {"url": frame}} for frame in frames],
        ],
    }


class OfficialContextBridge:
    """Clean official persisted model context before replies or compressors read it."""

    def __init__(self, database):
        self.database = database
        self.owners = []
        self.pending = {}
        self.locks = {}
        self.forwarding = ContextVar("meme_context_database_write", default=False)
        self.original_get = database.get_conversation_by_id
        self.original_update = database.update_conversation
        bridge = self

        async def update(
            cid, title=None, persona_id=None, content=None, token_usage=None
        ):
            # Stay below manager-level integrations: they may read context while
            # holding their own lock before or after forwarding this write.
            async with bridge.lock(cid):
                conv = await bridge.original_get(cid) if content is not None else None
                cleaned = await bridge.clean(conv.user_id if conv else "", cid, content)
                if cleaned != content:
                    token_usage = 0
                return await bridge.write(
                    cid,
                    title=title,
                    persona_id=persona_id,
                    content=cleaned,
                    token_usage=token_usage,
                )

        async def get(cid):
            # SQLite's native update returns a fresh row through this same
            # getter. It is already inside the write lock and already cleaned.
            if bridge.forwarding.get():
                return await bridge.original_get(cid)
            async with bridge.lock(cid):
                conv = await bridge.original_get(cid)
                if conv is None:
                    return None
                history = conv.content or []
                cleaned = await bridge.clean(conv.user_id, cid, history)
                if cleaned != history:
                    await bridge.write(cid, content=cleaned, token_usage=0)
                    return await bridge.original_get(cid)
                return conv

        self.get_wrapper, self.update_wrapper = get, update
        database.get_conversation_by_id = get
        database.update_conversation = update

    def lock(self, cid):
        return self.locks.setdefault(cid, asyncio.Lock())

    async def write(self, cid, **kwargs):
        token = self.forwarding.set(True)
        try:
            return await self.original_update(cid, **kwargs)
        finally:
            self.forwarding.reset(token)

    async def clean(self, origin, cid, history):
        owner = next(
            (o for o in reversed(self.owners) if o.config.get("enabled", True)), None
        )
        if history is None or owner is None:
            return history
        task = asyncio.current_task()
        self.pending.setdefault(owner, set()).add(task)
        try:
            return await owner.clean_history_context(origin, str(cid or ""), history)
        finally:
            self.pending.get(owner, set()).discard(task)

    @classmethod
    def acquire(cls, owner):
        manager = getattr(owner.context, "conversation_manager", None)
        database = getattr(manager, "db", None)
        if database is None:
            return None
        bridge = getattr(database, "_meme_history_bridge", None)
        if bridge is None:
            bridge = cls(database)
            database._meme_history_bridge = bridge
        bridge.owners.append(owner)
        return bridge

    async def close(self, owner):
        if owner in self.owners:
            self.owners.remove(owner)
        pending = self.pending.pop(owner, set()) - {asyncio.current_task()}
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.owners:
            return
        if self.database.get_conversation_by_id is self.get_wrapper:
            self.database.get_conversation_by_id = self.original_get
        if self.database.update_conversation is self.update_wrapper:
            self.database.update_conversation = self.original_update
        if getattr(self.database, "_meme_history_bridge", None) is self:
            delattr(self.database, "_meme_history_bridge")
