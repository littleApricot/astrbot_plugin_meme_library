"""Run an isolated dashboard fixture against the actual plugin Web API handlers."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import uvicorn
from astrbot.api import AstrBotConfig
from astrbot.api.web import PluginRequest, bind_request_context
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_meme_library.main import MemeLibrary

temp = tempfile.TemporaryDirectory(prefix="meme-library-ui-")
routes = {}


def register(route, handler, methods, description):
    routes[route.rsplit("/", 1)[1]] = handler


with patch(
    "astrbot_plugin_meme_library.main.StarTools.get_data_dir",
    return_value=Path(temp.name),
):
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
    )
    config = AstrBotConfig(str(Path(temp.name) / "config.json"), schema=schema)
    config.save_config(replace_config={"enabled": False})
    plugin = MemeLibrary(SimpleNamespace(register_web_api=register), config)

buffer = io.BytesIO()
Image.new("RGB", (300, 180), "#d7daf5").save(buffer, format="PNG")
record, _ = plugin.library.add(buffer.getvalue(), ".png", "test-group", "", "")
plugin.library.set_status(record["id"], "processing")
plugin.library.recognize(
    record["id"],
    "淡紫色图片",
    "测试图片：淡紫色背景。\n含义：用于测试说明编辑，不是真实群聊图片。",
    True,
)
app = FastAPI()


@app.api_route("/api/{name}", methods=["GET", "POST"])
async def api(name: str, request: Request):
    with bind_request_context(
        PluginRequest(
            request, plugin_name="astrbot_plugin_meme_library", username="local-test"
        )
    ):
        return await routes[name]()


@app.get("/")
async def page():
    source = (
        Path(__file__).resolve().parents[1] / "pages/library/index.html"
    ).read_text()
    bridge = """<script>window.AstrBotPluginPage={
      async apiGet(endpoint,params){if(/[?#]/.test(endpoint))throw Error('invalid bridge endpoint');const r=await fetch('/api/'+endpoint+'?'+new URLSearchParams(params||{}));const b=await r.json();if(!r.ok)throw Error(b.message||'failed');return b;},
      async apiPost(endpoint,body){const r=await fetch('/api/'+endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const b=await r.json();if(!r.ok)throw Error(b.message||'failed');return b;}
    };</script>"""
    return HTMLResponse(source.replace("<head>", "<head>" + bridge))


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 6196
    )
