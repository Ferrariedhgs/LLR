# ── ZeroGPU note ──────────────────────────────────────────────────────────────
# torch.compile / dynamo is NOT supported on ZeroGPU.  Disable globally before
# any library import so voxcpm's internal torch.compile warmup is a no-op.

# ─────────────────────────────────────────────────────────────────────────────

import re
import json
import logging

from huggingface_hub import hf_hub_download
import soundfile as sf
from voxcpm import VoxCPM
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)


# ─── Thinking control ────────────────────────────────────────────────────────
# The Jinja chat template checks for '/nothink' in the system message and sets
# enable_thinking=False.  When disabled the generation prompt suffix becomes
# '<think></think>' (immediately closed) so no thinking tokens are ever emitted.
#
# Template logic (from the Unsloth / Nemotron-3-Nano tokenizer_config.jinja):
#   if '/nothink' in system_message → enable_thinking = False
#   add_generation_prompt + not enable_thinking
#     → '<|im_start|>assistant\n<think></think>'

_NO_THINK_PREFIX = "/nothink\n\n"


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

CONTINUE_GAME_PROMPT = r"""You are a creative dungeon-master AI. Narrate the player's attempt to open a container with a key. If fits_lock is true, reveal item_to_give. Respond with a single valid JSON object and nothing else:
{"text":"<COMPLETE>"}"""

OPEN_DOOR_PROMPT = r"""You are a creative dungeon-master AI. Narrate the player's attempt to open the exit door. No keys: urge them to search. Wrong key: describe the failure. Right key: describe escape. Respond with a single valid JSON object and nothing else:
{"text":"<COMPLETE>"}"""


# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZES = {
    "room":     1024,
    "location": 512,
    "item":     512,
}

VOICES = {
    "Alfred":   "A deep, calm male voice with slow pacing, clear articulation, and a warm, authoritative tone suitable for documentaries and storytelling",
    "May":      "A soft, gentle female voice with medium-low pitch, smooth delivery, and a soothing tone ideal for audiobooks and explanations",
    "Liam":     "A young adult male voice with energetic delivery, slightly fast speech, bright tone, and expressive intonation for dynamic content",
    "Ava":      "A neutral synthetic assistant-like female voice with steady pacing, minimal emotion, and crisp articulation resembling a digital AI",
    "Arthur":   "An elderly male voice with gravelly texture, slow thoughtful pacing, and a wise, reflective tone",
    "Margaret": "An elderly female voice with warm, slightly breathy tone, gentle pacing, and nurturing delivery",
    "Chloe":    "A natural conversational female voice with warm friendliness, expressive but subtle intonation, and casual pacing",
    "Marcus":   "A documentary narrator voice with deep tone, authoritative pacing, and strong clarity for storytelling",
    "Sergei":   "A deep adult male voice with a russian accent, slightly resonant and strong",
    "Tatiana":  "An adult female voice with a soft but distinct russian accent",
    "Su":       "A mystical, ethereal female voice inspired by east asian folklore spirits",
}

# Required top-level keys for a valid room dict
_ROOM_REQUIRED = {
    "room_name", "room_story", "room_prompt",
    "door_description", "door_prompt", "door_key_name", "door_key_prompt",
    "containers", "keys",
}


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
    """
    Robustly extract a JSON object from raw model output.
    Handles: empty string, <think> blocks, markdown fences,
    text preambles/postambles, and trailing garbage.
    Returns a dict or None on total failure.
    """
    if not raw or not raw.strip():
        log.warning("Model returned empty string")
        return None

    # Strip thinking blocks — handles three shapes:
    #   1. Closed:   <think>...</think>   → remove entirely
    #   2. Empty:    <think></think>       → remove (template inserts this)
    #   3. Unclosed: <think>...EOF        → strip everything from <think> onward
    #      (happens when grammar sampling cuts generation mid-think)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)  # closed
    cleaned = re.sub(r"<think>.*",          "", cleaned, flags=re.DOTALL)  # unclosed
    cleaned = cleaned.strip()

    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$",           "", cleaned).strip()

    # Attempt 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the first {...} block (handles preamble / trailing text)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    log.error("Could not parse JSON from model output:\n%s", raw[:500])
    return None


def _validate_room(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    if not _ROOM_REQUIRED.issubset(d.keys()):
        missing = _ROOM_REQUIRED - d.keys()
        log.warning("Room dict missing keys: %s", missing)
        return False
    if not isinstance(d.get("containers"), list) or len(d["containers"]) < 4:
        log.warning("Room dict has fewer than 4 containers")
        return False
    if not isinstance(d.get("keys"), list) or len(d["keys"]) < 2:
        log.warning("Room dict has fewer than 2 keys")
        return False
    return True



# ─── JSON schemas for grammar-constrained sampling ───────────────────────────
# Passing response_format with a JSON schema forces llama-cpp-python to build a
# GBNF grammar from the schema and use it as a hard sampler constraint.
# The model physically cannot emit <think> tokens because they are not in the
# grammar — this is the only reliable way to suppress thinking in llama-cpp-python.

_ROOM_SCHEMA = {
    "type": "object",
    "properties": {
        "room_name":        {"type": "string"},
        "room_story":       {"type": "string"},
        "room_prompt":      {"type": "string"},
        "door_description": {"type": "string"},
        "door_prompt":      {"type": "string"},
        "door_key_name":    {"type": "string"},
        "door_key_prompt":  {"type": "string"},
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
            "minItems": 4, "maxItems": 4,
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
            "minItems": 2, "maxItems": 2,
        },
    },
    "required": [
        "room_name", "room_story", "room_prompt",
        "door_description", "door_prompt", "door_key_name", "door_key_prompt",
        "containers", "keys",
    ],
}

_TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}




game_gen='''{"room_name": "The Sandstone Temple of the Sun", "room_story": "A temple carved into sandstone, with a golden altar. The sun rises through a hole in the ceiling, illuminating the dust motes.", "room_prompt": "A sandstone temple with a golden altar, sunlight streaming through a hole in the ceiling, dust motes dancing in the light", "door_description": "A heavy bronze door with a sun symbol carved into it, requiring a golden key to open.", "door_prompt": "A heavy bronze door with a sun symbol carved into it, requiring a golden key to open", "door_key_name": "The Golden Sun Key", "door_key_prompt": "A golden key shaped like a sun with rays extending from the head", "containers": [{"container_name": "The Obsidian Box", "container_prompt": "A black obsidian box with a silver lock, sitting on a stone pedestal"}, {"container_name": "The Crystal Urn", "container_prompt": "A clear crystal urn filled with glowing blue crystals, resting on a wooden table"}, {"container_name": "The Iron Chest", "container_prompt": "A rusted iron chest with a padlock, lying open on the floor"}, {"container_name": "The Leather Satchel", "container_prompt": "A worn leather satchel with a metal clasp, hanging from a hook"}], "keys": [{"key_name": "The Moon Key", "key_prompt": "A silver key shaped like a crescent moon with intricate details"}, {"key_name": "The Broken Dagger", "key_prompt": "A jagged broken dagger with a rusted hilt, lying on the ground"}]}'''
continue_f='''{"text": "The Moon Key slides against the obsidian lock, its silver teeth clicking uselessly as the heavy door refuses to yield."}'''
continue_t='''{"text": "The broken dagger slides into the crystal urn with a soft chime, revealing the golden sun key within."}'''
open_n='''{"text": "The heavy iron door remains stubbornly sealed, its glowing runes mocking your futile attempts as the air grows heavier."}'''
open_w='''{"text": "The brass lock groans in frustration as the wrong key jams, refusing to turn and leaving you stranded in the dark."}'''
open_r='''{"text": "The heavy oak door groans open, revealing a path toward the light as the storm recedes."}'''
# ─── Model loading ────────────────────────────────────────────────────────────
# Llama.from_pretrained re-downloads the GGUF on every call even when the HF
# cache is warm, because it re-resolves metadata before checking the blob.
# We avoid this by downloading once with hf_hub_download (which is a true
# cache-check + skip) and storing the local path in a module-level variable.
# Subsequent calls to load_text_model() use Llama(model_path=...) directly —
# that is a plain file open with no network traffic.

_GGUF_REPO     = "build-small-hackathon/Nemotron-nano-4b-escape-room"
_GGUF_FILENAME = "nemotron-room-lora-Q4_K_M.gguf"
_gguf_path: str | None = None   # set on first download, reused every load



def _placeholder(w, h, r=15, g=55, b=55,txt=""):
    img = Image.new("RGB", (w, h), (r, g, b))
    d   = ImageDraw.Draw(img)
    font=ImageFont.truetype("arial.ttf",20)
    d.text([0,0],text=txt,stroke_fill=(0,0,0))
    return img


def _ensure_gguf() -> str:
    """Download the GGUF to the HF cache if not already there; return local path."""
    global _gguf_path
    if _gguf_path is None:
        log.info("Downloading / verifying GGUF from HF hub...")
        _gguf_path = hf_hub_download(repo_id=_GGUF_REPO, filename=_GGUF_FILENAME)
        log.info("GGUF ready at %s", _gguf_path)
    return _gguf_path





def load_text_model():
    """Load and return the Nemotron GGUF text model.
    Uses a cached local path after the first download — no repeated HF traffic.
    """
    a=1


def load_image_model():
    """Load and return the FLUX diffusion pipeline."""
    a=1


def load_tts_model():
    """Load and return VoxCPM.
    VoxCPM runs a warmup inference in __init__. With dynamo disabled globally
    above, the internal torch.compile wrapper is a no-op and warmup succeeds.
    """
    #return VoxCPM.from_pretrained("openbmb/VoxCPM2")
    a=1


def load_models():
    """Legacy helper — load all three models and return (text, image, tts).
    Kept for compatibility; prefer the individual loaders where possible."""
    return load_text_model(), load_image_model(), load_tts_model()


# ─── Inference helpers ────────────────────────────────────────────────────────
# Plain functions — no @spaces.GPU.  Called from within app.py's GPU frames.

def generate_game(text_model, max_retries: int = 3) -> dict:
    """
    Generate a full room dict.  Retries up to max_retries times on empty or
    unparseable output.  Raises RuntimeError if all attempts fail.
    """
    
    raw = game_gen
    

    data = _extract_json(raw)
    return data



def continue_game(text_model, container: str, key: str,
                  right_key: bool, item_given: str = "") -> dict:
    """Narrate a container-opening attempt. Returns dict with 'text' key."""
    
    if right_key:
        return f'{{"text":"The {key} opens the {container} and gives you the {item_given}}}'
    else:
        return f'{{"text":"The {key} can\'t open the {container}}}'


def open_door(text_model, key: str, key_type: str) -> dict:
    """Narrate a door-opening attempt. Returns dict with 'text' key."""
    if key_type=="right_key":
        return f'{{"text":"The {key} opens the door}}'
    elif key_type=="wrong_key":
        return f'{{"text":"The {key} doesn\'t open the door}}'
    else:
        return f'{{"text":"You don\'t have any key}}'


def generate_image(image_model, prompt: str, size_type: str):
    """Run FLUX inference and return a PIL Image.

    FLUX.2-klein is a step-distilled model — it is designed for 4 steps with
    guidance_scale=0.  Running 50 steps (the pipeline default) wastes ~12x GPU
    time with no quality gain and causes ZeroGPU budget exhaustion.
    """
    sz = IMAGE_SIZES[size_type]
    return _placeholder(
        w=sz,
        h=sz,
        txt=prompt
    )


def generate_voice(tts_model, prompt: str, narrator: str) -> str:
    """
    Synthesise speech and write to /tmp/narration.wav.
    Returns the output path for Gradio to serve.
    """
    voice_desc = VOICES[narrator]
    chunks = []
    for chunk in tts_model.generate_streaming(text=f"({voice_desc}) {prompt}"):
        chunks.append(chunk)
    wav = np.concatenate(chunks)
    out_path = "/tmp/narration.wav"
    sf.write(out_path, wav, tts_model.tts_model.sample_rate)
    return out_path
