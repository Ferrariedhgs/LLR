# ── Cache dirs — set before any library import ────────────────────────────────
import os
os.makedirs("/tmp/hf_cache",    exist_ok=True)
os.makedirs("/tmp/modelscope",  exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── ZeroGPU: disable torch.compile globally before any library import ─────────
import torch
import torchaudio
torch._dynamo.config.disable         = True
torch._dynamo.config.suppress_errors = True
torch.set_grad_enabled(False)
# ─────────────────────────────────────────────────────────────────────────────

import gc
import re
import json
import logging
import time

# transformers / diffusers / voxcpm are imported lazily inside each
# load_*() function so that CUDA is never initialised at module-import time.
# ZeroGPU requires all GPU-touching code to live inside @spaces.GPU frames.
from huggingface_hub import snapshot_download
import soundfile as sf
import numpy as np

log = logging.getLogger(__name__)
log.info("CUDA available: %s", torch.cuda.is_available())


# ─── Model repos ──────────────────────────────────────────────────────────────

_TEXT_REPO  = "build-small-hackathon/Nemotron-nano-4b-escape-room"
_TEXT_GGUF  = "nemotron-room-lora-Q4_K_M.gguf"  # only file we need at runtime
_IMAGE_REPO = "black-forest-labs/FLUX.2-klein-4B"
_TTS_REPO   = "openbmb/VoxCPM2"


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


# ─── Static failure messages ──────────────────────────────────────────────────

CONTAINER_FAIL_MESSAGES = [
    "Nothing happens.",
    "The key doesn't fit.",
    "You rattle the lock, but it holds firm.",
    "The container doesn't budge.",
    "That's not the right key.",
    "The lock refuses to yield.",
    "You try the key — it turns partway, then stops.",
    "A dull clunk. The lock stays shut.",
    "The mechanism doesn't respond.",
    "Something's missing. This isn't right.",
]

DOOR_FAIL_MESSAGES = [
    "The door won't open. You need the right key.",
    "You push and pull, but the door is firmly locked.",
    "Without the correct key, this door is going nowhere.",
    "The door stands firm. Keep searching.",
    "You don't have what's needed to open this door.",
    "The lock ignores you. Find the right key first.",
    "The handle doesn't move. The key is still out there.",
]


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
    "James":    "A professional British male voice with a refined accent, deep pitch, and confident delivery",
    "Victoria": "An elegant British female voice with clear diction, medium pitch, and graceful pacing",
    "Emma":     "A cheerful American female voice with bright energy, clear pronunciation, and expressive intonation",
    "Hiro":     "A calm Japanese male voice with a gentle accent, smooth delivery, and respectful tone",
}

# Required top-level keys for a valid room dict
_ROOM_REQUIRED = {
    "room_name", "room_story", "room_prompt",
    "door_description", "door_prompt", "door_key_name", "door_key_prompt",
    "containers", "keys",
}


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
    if not raw or not raw.strip():
        log.warning("Model returned empty string")
        return None

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*",          "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
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

    log.error("Could not parse JSON from model output:\n%s", raw[:500])
    return None


def _validate_room(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    if not _ROOM_REQUIRED.issubset(d.keys()):
        log.warning("Room dict missing keys: %s", _ROOM_REQUIRED - d.keys())
        return False
    if not isinstance(d.get("containers"), list) or len(d["containers"]) < 4:
        log.warning("Room dict has fewer than 4 containers")
        return False
    if not isinstance(d.get("keys"), list) or len(d["keys"]) < 2:
        log.warning("Room dict has fewer than 2 keys")
        return False
    return True


# ─── Weight pre-caching (warmup, CPU-only, no GPU lease) ──────────────────────

def _ensure_text_weights() -> None:
    """Pre-cache only the Q4_K_M GGUF file — no safetensors, no tokenizer,
    no mamba-ssm.  Returns the local path to the downloaded file."""
    log.info("_ensure_text_weights: downloading %s/%s…", _TEXT_REPO, _TEXT_GGUF)
    snapshot_download(
        repo_id=_TEXT_REPO,
        cache_dir="/tmp/hf_cache",
        allow_patterns=[_TEXT_GGUF],   # only the single GGUF we need
    )
    log.info("_ensure_text_weights: done.")


def _gguf_path() -> str:
    """Return the local filesystem path of the cached GGUF file.
    snapshot_download stores files under:
      <cache_dir>/models--<org>--<repo>/snapshots/<hash>/<filename>
    huggingface_hub.hf_hub_download resolves the latest snapshot for us.
    """
    from huggingface_hub import hf_hub_download as _hf_dl
    return _hf_dl(
        repo_id=_TEXT_REPO,
        filename=_TEXT_GGUF,
        cache_dir="/tmp/hf_cache",
    )


def _ensure_image_weights() -> None:
    log.info("_ensure_image_weights: downloading %s…", _IMAGE_REPO)
    snapshot_download(
        repo_id=_IMAGE_REPO,
        cache_dir="/tmp/hf_cache",
    )
    log.info("_ensure_image_weights: done.")


def _ensure_tts_weights() -> None:
    """Pre-cache VoxCPM2 — no CUDA init (would crash outside a GPU frame)."""
    log.info("_ensure_tts_weights: downloading %s…", _TTS_REPO)
    snapshot_download(
        repo_id=_TTS_REPO,
        cache_dir="/tmp/hf_cache",
    )
    log.info("_ensure_tts_weights: done.")


# ─── Chat template helper ─────────────────────────────────────────────────────
# llama-cpp-python applies the chat template stored in the GGUF metadata
# automatically via create_chat_completion(), so we no longer need a manual
# tokenizer.apply_chat_template() call.  The helper below is kept only as
# a thin compatibility shim in case we ever need the formatted string.

def _chatml_format(messages: list[dict]) -> str:
    """Manual ChatML fallback — only used for debugging; not called at runtime."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ─── Model loading / unloading ────────────────────────────────────────────────
# Each loader creates a fresh instance on GPU.
# Caller must call the matching unload_*() before loading the next model.

def load_text_model():
    """Load the Q4_K_M GGUF onto GPU via llama-cpp-python.
    Must be called from inside a @spaces.GPU decorated function so that CUDA
    is available when llama-cpp initialises its CUDA context.
    Returns a Llama instance (replaces the old (model, tokenizer) tuple).
    """
    log.info("load_text_model: loading GGUF %s from %s…", _TEXT_GGUF, _TEXT_REPO)
    # Lazy import keeps CUDA init inside the ZeroGPU frame.
    from llama_cpp import Llama as _Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    with open("nemotron-template.jinja") as f:
        template_str = f.read()
    
    formatter = Jinja2ChatFormatter(
        template=template_str,
        eos_token="<|im_end|>",
        bos_token="<|im_start|>",
    )
    chat_handler = formatter.to_chat_handler()

    gguf = _gguf_path()
    log.info("load_text_model: GGUF resolved to %s", gguf)
    llm = _Llama(
        model_path=gguf,
        n_gpu_layers=-1,        # offload every layer to CUDA
        n_ctx=2048,             # context window
        seed=int(time.time()),  # time-based seed for varied output
        verbose=False,
        chat_handler=chat_handler,
    )
    log.info("load_text_model: ready.")
    return llm


def unload_text_model(llm) -> None:
    log.info("unload_text_model: releasing…")
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("unload_text_model: done.")


def load_image_model():
    """Load FLUX diffusion pipeline onto GPU.
    Must be called from inside a @spaces.GPU decorated function.
    Returns pipeline.
    """
    log.info("load_image_model: loading %s onto GPU…", _IMAGE_REPO)
    from diffusers import DiffusionPipeline as _DiffusionPipeline
    pipe = _DiffusionPipeline.from_pretrained(
        _IMAGE_REPO,
        cache_dir="/tmp/hf_cache",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    log.info("load_image_model: ready.")
    return pipe


def unload_image_model(pipe) -> None:
    log.info("unload_image_model: releasing…")
    pipe.to("cpu")
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("unload_image_model: done.")


def load_tts_model():
    """Load VoxCPM2 onto GPU.
    Must be called from inside a @spaces.GPU decorated function.
    Returns VoxCPM instance.
    """
    log.info("load_tts_model: loading %s onto GPU…", _TTS_REPO)
    from voxcpm import VoxCPM as _VoxCPM
    model = _VoxCPM.from_pretrained(_TTS_REPO)
    log.info("load_tts_model: ready.")
    return model


def unload_tts_model(model) -> None:
    log.info("unload_tts_model: releasing…")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("unload_tts_model: done.")


# ─── Inference helpers ────────────────────────────────────────────────────────

def _generate_text(llm, messages: list[dict],
                   max_new_tokens: int = 1024, temperature: float = 0.8) -> str:
    """Single inference pass via llama-cpp create_chat_completion.
    llama-cpp applies the chat template stored in the GGUF metadata
    automatically, so we just pass the messages list directly.
    Returns the raw assistant content string.
    """
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


# ─── Game generation ──────────────────────────────────────────────────────────

def generate_game(llm, max_retries: int = 3) -> dict:
    """Generate a full room dict. Retries up to max_retries times."""
    for attempt in range(1, max_retries + 1):
        raw = _generate_text(
            llm,
            messages=[
                {"role": "system", "content": "/nothink\n\n"+GENERATE_GAME_PROMPT},
                {"role": "user",   "content": '{"task":"generate_room"}'},
            ],
            max_new_tokens=1024,
            temperature=0.8,
        )
        log.info("generate_game attempt %d raw output (%d chars): %s",
                 attempt, len(raw), raw[:200])
        data = _extract_json(raw)
        if data and _validate_room(data):
            return data
        log.warning("generate_game attempt %d failed validation, retrying…", attempt)

    raise RuntimeError(
        f"generate_game failed to produce valid JSON after {max_retries} attempts."
    )


def generate_image(image_model, prompt: str, size_type: str):
    """Run FLUX inference and return a PIL Image."""
    sz = IMAGE_SIZES[size_type]
    return image_model(
        prompt=prompt,
        width=sz,
        height=sz,
        num_inference_steps=4,
        guidance_scale=0.0,
    ).images[0]


def generate_voice(tts_model, prompt: str, narrator: str) -> tuple[np.ndarray, int]:
    """Synthesise speech. Returns (wav_array, sample_rate)."""
    if narrator not in VOICES:
        fallback = next(iter(VOICES))
        log.warning("generate_voice: unknown narrator %r — falling back to %r", narrator, fallback)
        narrator = fallback
    voice_desc = VOICES[narrator]
    wav = tts_model.generate(text=f"({voice_desc}) {prompt}")
    return wav, tts_model.tts_model.sample_rate


def _write_wav(wav: np.ndarray, sample_rate: int, path: str) -> str:
    sf.write(path, wav, sample_rate)
    return path


# ─── Pre-generation: narrative texts ─────────────────────────────────────────

def pregenerate_text(llm, room: dict, chain: list) -> dict:
    """
    Generate narrative texts for successful outcomes only:
      - correct open for each container (4 calls)
      - door success (1 call)

    Wrong-key / no-key failures use static messages at runtime.

    Returns:
        {
          "container_narrations": { box_idx: str, ... },
          "door_right":           str,
        }
    """
    log.info("pregenerate_text: generating correct-open narrations...")

    containers    = room["containers"]
    keys_list     = room["keys"]
    door_key_name = room.get("door_key_name", "the door key")
    ki_map        = {"key0": 0, "key1": 1}
    chain_map     = {e["box"]: e for e in chain}

    def _key_name(key_id: str) -> str:
        if key_id == "door_key":
            return door_key_name
        ki = ki_map.get(key_id)
        if ki is not None and ki < len(keys_list):
            return keys_list[ki].get("key_name", key_id)
        return key_id

    container_narrations: dict = {}

    for box_idx, container in enumerate(containers):
        container_name = container.get("container_name", f"Container {box_idx}")
        entry          = chain_map.get(box_idx, {})
        contains       = entry.get("contains")
        item_given     = _key_name(contains) if contains else ""

        raw = _generate_text(
            llm,
            messages=[
                {"role": "system", "content": CONTINUE_GAME_PROMPT},
                {"role": "user",   "content": (
                    f'{{"task":"continue_game","location_name":"{container_name}",'
                    f'"item_to_give":"{item_given}"}}'
                )},
            ],
            max_new_tokens=256,
            temperature=0.8,
        )
        data = _extract_json(raw)
        container_narrations[box_idx] = data.get("text", "") if data else ""
        log.info("pregenerate_text: container_%d done.", box_idx)

    log.info("pregenerate_text: generating door success narration...")
    raw = _generate_text(
        llm,
        messages=[
            {"role": "system", "content": OPEN_DOOR_PROMPT},
            {"role": "user",   "content": (
                f'{{"task":"open_door","given_key":"{door_key_name}"}}'
            )},
        ],
        max_new_tokens=256,
        temperature=0.8,
    )
    data = _extract_json(raw)
    door_right = data.get("text", "") if data else ""
    log.info("pregenerate_text: door_right done. Total: %d texts.",
             len(container_narrations) + 1)

    return {
        "container_narrations": container_narrations,
        "door_right":           door_right,
    }


# ─── Pre-generation: TTS audio ────────────────────────────────────────────────

def pregenerate_audio(tts_model, room: dict, chain: list,
                      narrations: dict, narrator: str) -> dict:
    """
    Generate audio for:
      - room_story
      - each correct container open
      - door success

    Returns:
        {
          "room_story":          "/tmp/audio_room_story.wav",
          "container_{box_idx}": "/tmp/audio_container_{box_idx}.wav",
          "door_right":          "/tmp/audio_door_right.wav",
        }
    """
    log.info("pregenerate_audio: synthesising audio clips...")
    audio_paths: dict = {}

    container_narrations = narrations["container_narrations"]
    door_right_text      = narrations["door_right"]

    story = room.get("room_story", "")
    if story:
        wav, sr = generate_voice(tts_model, story, narrator)
        audio_paths["room_story"] = _write_wav(wav, sr, "/tmp/audio_room_story.wav")
        log.info("pregenerate_audio: room_story done.")

    for box_idx, text in container_narrations.items():
        if text:
            wav, sr = generate_voice(tts_model, text, narrator)
            audio_paths[f"container_{box_idx}"] = _write_wav(
                wav, sr, f"/tmp/audio_container_{box_idx}.wav"
            )
            log.info("pregenerate_audio: container_%d done.", box_idx)

    if door_right_text:
        wav, sr = generate_voice(tts_model, door_right_text, narrator)
        audio_paths["door_right"] = _write_wav(wav, sr, "/tmp/audio_door_right.wav")
        log.info("pregenerate_audio: door_right done.")

    log.info("pregenerate_audio: all audio ready.")
    return audio_paths
