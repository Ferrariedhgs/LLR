import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TEXT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {
        "text": {"type": "string", "minLength": 1},
    },
}

SYSTEM_PROMPT = """You write short, atmospheric escape-room narration.
Return exactly one JSON object with a single "text" key containing the narration string.
Do not use markdown, comments, prose outside the JSON, XML tags, or analysis.
Do not include secret, API key, credential, token, or private information."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_continue_game_prompt(
    room: dict,
    container_name: str,
    key_name: str,
    fits_lock: bool,
    item_to_give: str | None,
) -> str:
    input_obj: dict = {
        "task": "continue_game",
        "location_name": container_name,
        "key_name": key_name,
        "fits_lock": fits_lock,
    }
    if fits_lock and item_to_give:
        input_obj["item_to_give"] = item_to_give

    if fits_lock:
        instruction = (
            f'The key "{key_name}" fits the "{container_name}" and opens it, '
            f'revealing "{item_to_give}" inside. '
            "Describe the key working and the player finding the item."
        )
    else:
        instruction = (
            f'The key "{key_name}" does not open "{container_name}". '
            "Describe the failure. Vary your phrasing — avoid always saying 'doesn't fit'; "
            "use different ways to express that the key doesn't work."
        )

    return f"""/no_think
You are writing narration for an escape room game.

Room: {room["room_name"]}
Atmosphere: {room["room_story"]}

{instruction}

Write one sentence, 10-25 words, second-person perspective, atmospheric.

Input the player sent:
{json.dumps(input_obj, indent=2)}

Return exactly one JSON object:
{{"text": "<narration here>"}}

Output only valid JSON."""


def build_open_door_prompt(room: dict, has_key: str, given_key: str) -> str:
    input_obj = {
        "task": "open_door",
        "has_key": has_key,
        "given_key": given_key,
    }

    if has_key == "right_key":
        instruction = (
            f'The player uses "{given_key}" which is the correct key. '
            "The door opens. Write a short triumphant sentence hinting at escape."
        )
    elif has_key == "wrong_key":
        instruction = (
            f'The player tries "{given_key}" but it is the wrong key. '
            "The door stays locked. Vary your phrasing across different rooms."
        )
    else:  # no_key
        instruction = (
            "The player has no key at all. "
            "The door stays locked. Vary your phrasing across different rooms."
        )

    return f"""/no_think
You are writing narration for an escape room game.

Room: {room["room_name"]}
Door: {room["door_description"]}
Atmosphere: {room["room_story"]}

{instruction}

Write one sentence, 10-25 words, second-person perspective, atmospheric.

Input the player sent:
{json.dumps(input_obj, indent=2)}

Return exactly one JSON object:
{{"text": "<narration here>"}}

Output only valid JSON."""


# ---------------------------------------------------------------------------
# Interaction row generators
# ---------------------------------------------------------------------------

def generate_continue_game_rows(room: dict, rng: random.Random, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []

    containers = room.get("containers", [])
    all_keys = _collect_all_key_names(room)

    if not all_keys:
        print(f"  Warning: room '{room.get('room_name')}' has no keys, skipping continue_game", file=sys.stderr)
        return rows

    # Shuffle a copy of the key pool and assign one "correct" key per container.
    # Keys are picked without replacement where possible, then cycle if needed.
    shuffled = list(all_keys)
    rng.shuffle(shuffled)
    assigned_correct = [shuffled[i % len(shuffled)] for i in range(len(containers))]

    for container, correct_key in zip(containers, assigned_correct):
        c_name = container["container_name"]
        fits_lock = rng.random() < args.true_ratio

        if fits_lock:
            key_name = correct_key
            # item_to_give: any key from the pool except the one being used
            candidates = [k for k in all_keys if k.lower() != correct_key.lower()]
            item_to_give = rng.choice(candidates) if candidates else correct_key
            instruction = {
                "task": "continue_game",
                "location_name": c_name,
                "key_name": key_name,
                "fits_lock": True,
                "item_to_give": item_to_give,
            }
            prompt = build_continue_game_prompt(room, c_name, key_name, True, item_to_give)
        else:
            # Pick any key that isn't the assigned correct one for this container
            wrong_candidates = [k for k in all_keys if k.lower() != correct_key.lower()]
            key_name = rng.choice(wrong_candidates) if wrong_candidates else correct_key
            instruction = {
                "task": "continue_game",
                "location_name": c_name,
                "key_name": key_name,
                "fits_lock": False,
            }
            prompt = build_continue_game_prompt(room, c_name, key_name, False, None)

        text = _call_with_retries(prompt, args, label=f"continue_game [{c_name}]")
        if text:
            rows.append(_make_row(instruction, {"text": text}))

    return rows


def generate_open_door_rows(room: dict, rng: random.Random, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    door_key = room["door_key_name"]
    all_keys = _collect_all_key_names(room)
    wrong_candidates = [k for k in all_keys if k.lower() != door_key.lower()]

    # Randomly pick one of the three states
    has_key = rng.choice(["no_key", "wrong_key", "right_key"])

    if has_key == "right_key":
        given_key = door_key
    elif has_key == "wrong_key":
        given_key = rng.choice(wrong_candidates) if wrong_candidates else door_key
    else:
        given_key = "none"

    instruction = {
        "task": "open_door",
        "has_key": has_key,
        "given_key": given_key,
    }
    prompt = build_open_door_prompt(room, has_key, given_key)
    text = _call_with_retries(prompt, args, label=f"open_door [{has_key}]")
    if text:
        rows.append(_make_row(instruction, {"text": text}))

    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_all_key_names(room: dict) -> list[str]:
    """Collect every named key in the room. container_key_name is optional."""
    names: list[str] = [room["door_key_name"]]
    for c in room.get("containers", []):
        if "container_key_name" in c:
            names.append(c["container_key_name"])
    for k in room.get("keys", []):
        names.append(k["key_name"])
    return names


def _make_row(instruction: dict, response: dict) -> dict:
    return {
        "instruction": json.dumps(instruction, separators=(",", ":")),
        "response": json.dumps(response, separators=(",", ":")),
    }


def extract_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start: end + 1])


def validate_text_response(data: dict) -> str:
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("response missing non-empty 'text' field")
    return text.strip()


def _call_with_retries(prompt: str, args: argparse.Namespace, label: str) -> str | None:
    for attempt in range(1, args.attempts + 1):
        try:
            raw = call_model(prompt, args)
            text = validate_text_response(extract_json_object(raw))
            return text
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  [{label}] retry {attempt}/{args.attempts}: {exc}", file=sys.stderr, flush=True)
    print(f"  [{label}] skipped after {args.attempts} retries", file=sys.stderr, flush=True)
    return None


# ---------------------------------------------------------------------------
# Ollama I/O
# ---------------------------------------------------------------------------

def build_ollama_payload(prompt: str, args: argparse.Namespace) -> dict:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repeat_penalty": args.repeat_penalty,
            "num_predict": args.max_tokens,
            "num_ctx": args.ctx_size,
            "seed": args.seed,
        },
    }
    if args.json_mode:
        payload["format"] = TEXT_RESPONSE_SCHEMA
    return payload


def post_json(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = None
    method = "GET"
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_ollama_chat(payload: dict, args: argparse.Namespace) -> dict:
    try:
        return post_json(f"{args.ollama_host}/api/chat", payload, timeout=300)
    except urllib.error.URLError as exc:
        if isinstance(exc, urllib.error.HTTPError):
            if exc.code == 404:
                raise SystemExit(
                    f"Model '{args.model}' not found in Ollama.\n"
                    f"Pull it first with: ollama pull {args.model}"
                )
            raise
        raise SystemExit(
            f"Could not connect to Ollama at {args.ollama_host}\n"
            "Make sure Ollama is running: ollama serve"
        )


def call_model(prompt: str, args: argparse.Namespace) -> str:
    payload = build_ollama_payload(prompt, args)
    try:
        data = post_ollama_chat(payload, args)
    except urllib.error.HTTPError as exc:
        if args.json_mode and exc.code == 400:
            # Schema not supported — fall back to plain json mode
            saved = args.json_mode
            args.json_mode = False
            payload = build_ollama_payload(prompt, args)
            args.json_mode = saved
            data = post_ollama_chat(payload, args)
        else:
            raise

    message = data.get("message")
    content = message.get("content", "") if isinstance(message, dict) else data.get("response", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"model returned empty content")
    return content


# ---------------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------------

def load_rooms(path: Path) -> list[dict]:
    rooms: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Rooms may be stored wrapped as {"instruction":..., "response":...}
                # or as bare room objects — handle both.
                if "response" in obj and "instruction" in obj:
                    obj = json.loads(obj["response"])
                rooms.append(obj)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping line {line_num} in rooms file: {exc}", file=sys.stderr)
    return rooms


def write_rows(path: Path, rows: list[dict], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush_batch(path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    write_rows(path, rows, append=True)
    count = len(rows)
    rows.clear()
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_dataset(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    rooms_path = Path(args.rooms_file)

    if not rooms_path.exists():
        raise SystemExit(f"Rooms file not found: {rooms_path}")

    rooms = load_rooms(rooms_path)
    if not rooms:
        raise SystemExit("No rooms loaded — check your rooms file.")
    print(f"Loaded {len(rooms)} rooms from {rooms_path}", flush=True)

    # Verify Ollama is reachable
    try:
        tags = post_json(f"{args.ollama_host}/api/tags", timeout=10)
        available_models = [m["name"] for m in tags.get("models", [])]
        model_available = any(
            m == args.model or m.startswith(args.model + ":") or args.model.startswith(m.split(":")[0])
            for m in available_models
        )
        if not model_available:
            print(
                f"Warning: model '{args.model}' not found in Ollama. "
                f"Available: {available_models}\n"
                f"Pull it with: ollama pull {args.model}",
                file=sys.stderr,
            )
    except urllib.error.URLError:
        raise SystemExit(
            f"Could not connect to Ollama at {args.ollama_host}\n"
            "Make sure Ollama is running: ollama serve"
        )

    pending: list[dict] = []
    written_count = 0

    for room_index, room in enumerate(rooms):
        room_name = room.get("room_name", f"room_{room_index + 1}")
        print(f"[{room_index + 1}/{len(rooms)}] {room_name}", flush=True)

        new_rows: list[dict] = []
        new_rows += generate_continue_game_rows(room, rng, args)
        new_rows += generate_open_door_rows(room, rng, args)

        pending.extend(new_rows)
        print(f"  → {len(new_rows)} rows (pending: {len(pending)})", flush=True)

        if len(pending) >= args.batch_size:
            written_count += flush_batch(output_path, pending)
            print(f"  Saved {written_count} total rows to {args.output}", flush=True)

    if pending:
        written_count += flush_batch(output_path, pending)

    return written_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate escape-room narration rows from an existing rooms JSONL file.\n"
            "Each room produces one continue_game row per container + one open_door row."
        )
    )
    parser.add_argument(
        "--rooms-file",
        default="F:/Projects/datasets/Nemotron-escape/generate-room-dataset.jsonl",
        help="Path to the existing rooms JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="F:/Projects/datasets/Nemotron-escape/answer-dataset.jsonl",
        help="Output dataset path.",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:4b",
        help="Ollama model name. Must be pulled first.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API base URL.",
    )
    parser.add_argument(
        "--true-ratio",
        type=float,
        default=0.3,
        help="Probability that a continue_game sample will be fits_lock=True (default 0.5).",
    )
    parser.add_argument("--batch-size", type=int, default=25, help="Save every N pending rows.")
    parser.add_argument("--attempts", type=int, default=5, help="Retries per LLM call.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repeat-penalty", type=float, default=1.08)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-json-mode",
        dest="json_mode",
        action="store_false",
        help="Disable Ollama structured JSON mode.",
    )
    parser.set_defaults(json_mode=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_count = generate_dataset(args)
    print(f"Wrote {written_count} rows to {args.output}")


if __name__ == "__main__":
    main()
