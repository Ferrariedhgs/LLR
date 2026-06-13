"""
1000 Rooms — Escape-Room game backend
Served via gradio.Server (FastAPI + Gradio queuing + ZeroGPU support).
"""

import os
import time
import re
import json
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
import spaces
from gradio import Server
from fastapi.responses import HTMLResponse

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─── Constants ────────────────────────────────────────────────────────────────

_NO_THINK_PREFIX = "/nothink\n\n"
_GGUF_REPO       = "build-small-hackathon/Nemotron-nano-4b-escape-room"
_GGUF_FILENAME   = "nemotron-room-lora-Q4_K_M.gguf"

_ROOM_REQUIRED = {
    "room_name", "room_story", "room_prompt",
    "door_description", "door_prompt", "door_key_name", "door_key_prompt",
    "containers", "keys",
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

GENERATE_GAME_PROMPT = r"""You are a creative dungeon-master AI that generates escape-room content. Always respond with a single valid JSON object and nothing else — no markdown, no explanation, no preamble. Complete every <COMPLETE> placeholder:
{"room_name":"<COMPLETE>",
"room_story":"<COMPLETE>",
"room_prompt":"<COMPLETE>",
"door_description":"<COMPLETE>",
"door_prompt":"<COMPLETE>",
"door_key_name":"<COMPLETE>",
"door_key_prompt":"<COMPLETE>",
"containers":[
{"container_name":"<COMPLETE>","container_prompt":"<COMPLETE>"},
{"container_name":"<COMPLETE>","container_prompt":"<COMPLETE>"},
{"container_name":"<COMPLETE>","container_prompt":"<COMPLETE>"},
{"container_name":"<COMPLETE>","container_prompt":"<COMPLETE>"}
],
"keys":[
{"key_name":"<COMPLETE>","key_prompt":"<COMPLETE>"},
{"key_name":"<COMPLETE>","key_prompt":"<COMPLETE>"}
]}"""

CONTINUE_GAME_PROMPT = r"""You are a creative dungeon-master AI. The player has found the correct key and opened a container. Narrate the discovery vividly — describe opening the container and reveal item_to_give. Respond with a single valid JSON object and nothing else:
{"text":"<COMPLETE>"}"""

OPEN_DOOR_PROMPT = r"""You are a creative dungeon-master AI. The player has used the correct key to open the exit door. Describe their triumphant escape in vivid detail. Respond with a single valid JSON object and nothing else:
{"text":"<COMPLETE>"}"""

# ─── JSON schema for constrained sampling ─────────────────────────────────────

_ROOM_SCHEMA = {
    "type": "object",
    "properties": {
        "room_name":         {"type": "string"},
        "room_story":        {"type": "string"},
        "room_prompt":       {"type": "string"},
        "door_description":  {"type": "string"},
        "door_prompt":       {"type": "string"},
        "door_key_name":     {"type": "string"},
        "door_key_prompt":   {"type": "string"},
        "containers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "container_name":   {"type": "string"},
                    "container_prompt": {"type": "string"},
                },
                "required": ["container_name", "container_prompt"],
            },
            "minItems": 4,
        },
        "keys": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key_name":   {"type": "string"},
                    "key_prompt": {"type": "string"},
                },
                "required": ["key_name", "key_prompt"],
            },
            "minItems": 2,
        },
    },
    "required": list(_ROOM_REQUIRED),
}

_TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

# ─── Model loading ─────────────────────────────────────────────────────────────

_model_path: str | None = None


def _ensure_gguf() -> str:
    global _model_path
    if _model_path and Path(_model_path).exists():
        return _model_path
    log.info("Downloading GGUF from HF Hub…")
    _model_path = hf_hub_download(repo_id=_GGUF_REPO, filename=_GGUF_FILENAME)
    log.info("GGUF cached at %s", _model_path)
    return _model_path


def load_text_model():
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    log.info("load_text_model: loading onto GPU…")
    with open("nemotron-template.jinja") as f:
        template_str = f.read()
    formatter = Jinja2ChatFormatter(
        template=template_str,
        eos_token="<|im_end|>",
        bos_token="<|im_start|>",
    )
    chat_handler = formatter.to_chat_handler()
    model_path = _ensure_gguf()
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        #flash_attn=True,
        seed=int(time.time()),  # time-based seed for varied output
        verbose=False,
        chat_handler=chat_handler,
    )
    log.info("load_text_model: GPU model ready.")
    return model


# ─── JSON helpers ──────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
    if not raw or not raw.strip():
        log.warning("Model returned empty string")
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*",          "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$",           "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    log.error("Could not parse JSON:\n%s", raw[:500])
    return None


def _validate_room(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    if not _ROOM_REQUIRED.issubset(d.keys()):
        log.warning("Room missing keys: %s", _ROOM_REQUIRED - d.keys())
        return False
    if not isinstance(d.get("containers"), list) or len(d["containers"]) < 4:
        log.warning("Room has fewer than 4 containers")
        return False
    if not isinstance(d.get("keys"), list) or len(d["keys"]) < 2:
        log.warning("Room has fewer than 2 keys")
        return False
    return True


# ─── Generation helpers ───────────────────────────────────────────────────────

def _chat(model, system: str, user: str, schema: dict, max_tokens: int = 1024) -> dict | None:
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": _NO_THINK_PREFIX + system},
            {"role": "user",   "content": user},
        ],
        response_format={"type": "json_object", "schema": schema},
        temperature=0.8,
        max_tokens=max_tokens,
    )
    raw = response["choices"][0]["message"]["content"]
    return _extract_json(raw)


def generate_game(model, max_retries: int = 3) -> dict:
    for attempt in range(1, max_retries + 1):
        data = _chat(model, GENERATE_GAME_PROMPT, '{"task":"generate_room"}', _ROOM_SCHEMA)
        if data and _validate_room(data):
            return data
        log.warning("generate_game attempt %d failed, retrying…", attempt)
    raise RuntimeError(f"generate_game failed after {max_retries} attempts.")


def continue_game(model, container_name: str, key_name: str) -> str:
    payload = json.dumps({"container_name": container_name, "item_to_give": key_name})
    data = _chat(model, CONTINUE_GAME_PROMPT, payload, _TEXT_SCHEMA)
    return data["text"] if data else f"You open the {container_name} and find the {key_name}."


def open_door(model, room_name: str, door_name: str) -> str:
    payload = json.dumps({"room_name": room_name, "door_name": door_name})
    data = _chat(model, OPEN_DOOR_PROMPT, payload, _TEXT_SCHEMA)
    return data["text"] if data else "You escape! The door swings open and freedom awaits."


# ─── App ──────────────────────────────────────────────────────────────────────

app = Server()
_text_model = None


@app.get("/")
async def homepage():
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.post("/generate_room")
@spaces.GPU(duration=180)
def api_generate_room():
    global _text_model
    if _text_model is None:
        _text_model = load_text_model()
    return generate_game(_text_model)


@app.post("/continue_room")
@spaces.GPU(duration=120)
def api_continue_room(container_name: str, key_name: str):
    global _text_model
    if _text_model is None:
        _text_model = load_text_model()
    text = continue_game(_text_model, container_name, key_name)
    return {"text": text}


@app.post("/open_door")
@spaces.GPU(duration=120)
def api_open_door(room_name: str, door_name: str):
    global _text_model
    if _text_model is None:
        _text_model = load_text_model()
    text = open_door(_text_model, room_name, door_name)
    return {"text": text}


if __name__ == "__main__":
    app.launch(show_error=True)
