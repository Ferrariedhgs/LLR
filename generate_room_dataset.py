import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


INSTRUCTION = {"task": "genearate_room"}

REQUIRED_TOP_LEVEL_KEYS = [
    "room_name",
    "room_story",
    "room_prompt",
    "door_description",
    "door_prompt",
    "door_key_name",
    "door_key_prompt",
    "containers",
    "keys",
]

ROOM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": REQUIRED_TOP_LEVEL_KEYS,
    "properties": {
        "room_name": {"type": "string", "minLength": 1},
        "room_story": {"type": "string", "minLength": 1},
        "room_prompt": {"type": "string", "minLength": 1},
        "door_description": {"type": "string", "minLength": 1},
        "door_prompt": {"type": "string", "minLength": 1},
        "door_key_name": {"type": "string", "minLength": 1},
        "door_key_prompt": {"type": "string", "minLength": 1},
        "containers": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["container_name", "container_prompt"],
                "properties": {
                    "container_name": {"type": "string", "minLength": 1},
                    "container_prompt": {"type": "string", "minLength": 1},
                },
            },
        },
        "keys": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key_name", "key_prompt"],
                "properties": {
                    "key_name": {"type": "string", "minLength": 1},
                    "key_prompt": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You generate synthetic escape-room training examples.
Return exactly one JSON object that matches the requested schema.
Do not use markdown, comments, prose, XML tags, or analysis.
Do not include secret, API key, credential, token, or private information."""

THEMES = [
    "abandoned arctic research station",
    "sunken chapel",
    "circus tent",
    "art deco hotel service floor",
    "desert observatory",
    "haunted puppet theater",
    "overgrown greenhouse laboratory",
    "clockmaker's attic",
    "underground metro control room",
    "moon mining barracks",
    "forgotten royal bathhouse",
    "shipwreck captain's cabin",
    "neon arcade after closing",
    "medieval plague infirmary",
    "volcanic forge",
    "misty lighthouse keeper room",
    "silent film studio prop room",
    "ancient library vault",
    "alien botanical quarantine",
    "snowbound hunting lodge",
    "subterranean mushroom farm",
    "carnival fortune teller wagon",
    "old radio broadcast booth",
    "clock tower bell chamber",
    "industrial laundry basement",
    "museum restoration workshop",
    "samurai armory",
    "pirate map archive",
    "Victorian seance parlor",
    "cyberpunk noodle shop back room",
    "bone-dry wine cellar",
    "ruined planetarium",
    "jungle temple antechamber",
    "opera house costume storage",
    "deep-sea diving locker room",
    "wizard school detention room",
    "derelict space elevator station",

    # Sci-fi
    "crashed alien scout ship",
    "generation ship hydroponics bay",
    "orbital defense command center",
    "robot repair workshop",
    "terraforming control bunker",
    "cryogenic storage facility",
    "quantum computing laboratory",
    "asteroid refinery office",
    "deep-space navigation bridge",
    "android memory archive",
    "interstellar customs checkpoint",
    "abandoned cloning facility",
    "wormhole observation deck",
    "fusion reactor maintenance tunnel",
    "spaceport cargo warehouse",
    "alien embassy waiting room",
    "nanotechnology research lab",
    "cybernetic surgery theater",
    "rogue AI server chamber",
    "time travel calibration room",

    # Fantasy
    "dragon keeper's treasury",
    "enchanted apothecary",
    "abandoned mage tower",
    "goblin engineering workshop",
    "druid ritual grove",
    "phoenix sanctuary",
    "crystal cavern shrine",
    "royal alchemist laboratory",
    "cursed throne room",
    "giant's pantry",
    "elven star observatory",
    "witch's cottage cellar",
    "griffin stable",
    "necromancer crypt",
    "fairy queen audience chamber",
    "floating castle engine room",
    "runic archive hall",
    "forgotten dungeon warden office",
    "elemental summoning chamber",
    "frozen sorcerer sanctuary",

    # Historical
    "Roman bath maintenance room",
    "Renaissance inventor workshop",
    "Napoleonic war map room",
    "medieval monastery scriptorium",
    "Viking longhouse treasury",
    "ancient Egyptian embalming chamber",
    "Byzantine archive room",
    "wild west sheriff office",
    "prohibition-era speakeasy cellar",
    "Victorian inventor laboratory",
    "feudal Japanese tea house",
    "Ottoman astronomer observatory",
    "royal cartography office",
    "castle siege supply depot",
    "medieval treasury vault",
    "ancient Greek oracle chamber",
    "colonial trading company office",
    "first world war field bunker",
    "steam locomotive dispatch office",
    "royal crown jeweler workshop",

    # Horror
    "abandoned asylum records room",
    "bloodstained butcher shop",
    "cursed doll workshop",
    "forgotten mortuary",
    "occult ritual basement",
    "haunted hotel suite",
    "underground catacomb chapel",
    "vampire noble study",
    "werewolf hunter lodge",
    "blackout hospital ward",
    "abandoned carnival ride control room",
    "shadow cult archive",
    "fog-covered fishing shack",
    "possessed artist studio",
    "undertaker preparation room",
    "haunted orphanage classroom",
    "creaking funeral parlor",
    "crypt beneath a ruined church",
    "taxidermist workshop",
    "abandoned prison isolation cell",

    # Steampunk
    "airship engine compartment",
    "steam-powered automaton factory",
    "clockwork submarine cabin",
    "victorian electrical laboratory",
    "brass observatory dome",
    "mechanical post office sorting room",
    "steam rail control station",
    "aether generator room",
    "inventor guild workshop",
    "clockwork parliament archive",

    # Modern / Realistic
    "abandoned shopping mall security office",
    "mountain fire lookout station",
    "luxury casino surveillance room",
    "airport baggage control center",
    "hotel laundry facility",
    "university chemistry lab",
    "escape room designer workshop",
    "underground parking security hub",
    "bank records archive",
    "broadcast television control room",
    "power plant monitoring room",
    "weather station operations center",
    "data center server aisle",
    "forensic evidence storage room",
    "museum security office",
    "cargo ship communications room",
    "mountain rescue headquarters",
    "film prop warehouse",
    "subway maintenance depot",
    "private investigator office",

    # Nature
    "hidden waterfall cave",
    "ancient redwood ranger station",
    "glacier expedition camp",
    "volcanic research shelter",
    "desert caravan resting tent",
    "coral reef diving station",
    "mountain monastery library",
    "jungle canopy observation platform",
    "ice cave shrine",
    "forest fire watchtower",

    # Mystery
    "locked detective evidence room",
    "secret society meeting hall",
    "missing explorer study",
    "hidden railway station office",
    "abandoned telegram center",
    "mysterious auction house vault",
    "forgotten safehouse",
    "encrypted records archive",
    "private museum collection room",
    "eccentric collector mansion wing",

    # Weird / Surreal
    "upside-down apartment",
    "room trapped inside a painting",
    "infinite hotel corridor junction",
    "dream architect workshop",
    "gravity-shift testing chamber",
    "living clock interior",
    "library between dimensions",
    "colorless artist studio",
    "museum of impossible objects",
    "room inside a giant mechanical heart",
    "memory reconstruction laboratory",
    "train car traveling through dreams",
    "floating island weather station",
    "hallway looping through time",
    "house built inside a whale skeleton",
    "cosmic lighthouse beyond reality",
    "abandoned simulation debugging room",
    "archive of forgotten memories",
    "mirror dimension observatory",
    "door factory for alternate worlds"
]


def build_prompt(theme: str, used_room_names: list[str]) -> str:
    used_text = ", ".join(used_room_names[-20:]) or "none"
    template = {
        "room_name": "Short evocative room name",
        "room_story": "Second-person escape room setup. Mention the locked door, atmosphere, and urgency.",
        "room_prompt": "Visual prompt for the whole room",
        "door_description": "Second-person description of the door and its lock symbol",
        "door_prompt": "Visual prompt for the locked door",
        "door_key_name": "Name of the one correct key for the door",
        "door_key_prompt": "Visual prompt for the correct door key",
        "containers": [
            {"container_name": "Container 1", "container_prompt": "Visual prompt"},
            {"container_name": "Container 2", "container_prompt": "Visual prompt"},
            {"container_name": "Container 3", "container_prompt": "Visual prompt"},
            {"container_name": "Container 4", "container_prompt": "Visual prompt"},
        ],
        "keys": [
            {"key_name": "Wrong key or special object 1", "key_prompt": "Visual prompt"},
            {"key_name": "Wrong key or special object 2", "key_prompt": "Visual prompt"},
        ],
    }

    return f"""/no_think
Create one synthetic escape-room JSON object.

Return one valid JSON object only. Do not use markdown. Do not add comments.
The JSON must match this exact schema and key order:
{json.dumps(template, indent=2)}

Rules:
- Theme: {theme}
- Always create exactly 4 containers.
- Always create exactly 2 keys in the "keys" array.
- "door_key_name" is the correct key for the door.
- The 2 keys in the "keys" array are decoys or intermediate keys and must not duplicate "door_key_name".
- Use varied containers and varied key-like items that fit the theme.
- Make every string non-empty.
- Avoid these already used room names: {used_text}
- Output only valid JSON."""


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
        return json.loads(text[start : end + 1])


def require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def validate_room(room: dict) -> dict:
    if not isinstance(room, dict):
        raise ValueError("room must be a JSON object")

    clean = {}
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in room:
            raise ValueError(f"missing key: {key}")

    for key in REQUIRED_TOP_LEVEL_KEYS[:7]:
        clean[key] = require_string(room[key], key)

    containers = room["containers"]
    if not isinstance(containers, list) or len(containers) != 4:
        raise ValueError("containers must contain exactly 4 objects")

    clean_containers = []
    seen_containers = set()
    for index, container in enumerate(containers):
        if not isinstance(container, dict):
            raise ValueError(f"container {index + 1} must be an object")
        name = require_string(container.get("container_name"), f"container {index + 1}.container_name")
        prompt = require_string(container.get("container_prompt"), f"container {index + 1}.container_prompt")
        lowered = name.lower()
        if lowered in seen_containers:
            raise ValueError(f"duplicate container name: {name}")
        seen_containers.add(lowered)
        clean_containers.append({"container_name": name, "container_prompt": prompt})

    keys = room["keys"]
    if not isinstance(keys, list) or len(keys) != 2:
        raise ValueError("keys must contain exactly 2 objects")

    clean_keys = []
    seen_keys = {clean["door_key_name"].lower()}
    for index, key_obj in enumerate(keys):
        if not isinstance(key_obj, dict):
            raise ValueError(f"key {index + 1} must be an object")
        name = require_string(key_obj.get("key_name"), f"key {index + 1}.key_name")
        prompt = require_string(key_obj.get("key_prompt"), f"key {index + 1}.key_prompt")
        lowered = name.lower()
        if lowered in seen_keys:
            raise ValueError(f"duplicate or door-key key name: {name}")
        seen_keys.add(lowered)
        clean_keys.append({"key_name": name, "key_prompt": prompt})

    clean["containers"] = clean_containers
    clean["keys"] = clean_keys
    return clean


def response_summary(data: dict) -> str:
    safe_keys = [
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "eval_count",
        "error",
    ]
    summary = {key: data.get(key) for key in safe_keys if key in data}
    message = data.get("message")
    if isinstance(message, dict):
        summary["message_keys"] = sorted(message.keys())
        content = message.get("content")
        summary["message_content_length"] = len(content) if isinstance(content, str) else 0
    response = data.get("response")
    if isinstance(response, str):
        summary["response_length"] = len(response)
    return json.dumps(summary, ensure_ascii=True)


def build_ollama_payload(prompt: str, args: argparse.Namespace, use_schema: bool) -> dict:
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
        payload["format"] = ROOM_SCHEMA if use_schema else "json"

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
        return post_json(
            f"{args.ollama_host}/api/chat",
            payload,
            timeout=300,
        )
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
    payload = build_ollama_payload(prompt, args, use_schema=True)

    try:
        data = post_ollama_chat(payload, args)
    except urllib.error.HTTPError as exc:
        if args.json_mode and exc.code == 400:
            payload = build_ollama_payload(prompt, args, use_schema=False)
            data = post_ollama_chat(payload, args)
        else:
            raise

    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = data.get("response", "")

    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"model returned empty content: {response_summary(data)}")

    return content


def write_dataset(path: Path, rows: list[dict], output_format: str, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "array":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush_batch(path: Path, rows: list[dict], output_format: str) -> int:
    if not rows:
        return 0
    write_dataset(path, rows, output_format, append=True)
    count = len(rows)
    rows.clear()
    return count


def generate_dataset(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    pending_rows = []
    written_count = 0
    skipped_count = 0
    used_room_names = []

    if args.format == "array" and output_path.exists():
        raise SystemExit(
            "Appending to --format array is not supported. Use --format jsonl for appendable batch writes."
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    # Verify Ollama is reachable and the model exists before starting
    try:
        tags = post_json(f"{args.ollama_host}/api/tags", timeout=10)
        available_models = [m["name"] for m in tags.get("models", [])]
        # Ollama model names may include a tag like "qwen2.5:7b"; check prefix match too.
        model_available = any(
            m == args.model or m.startswith(args.model + ":") or args.model.startswith(m.split(":")[0])
            for m in available_models
        )
        if not model_available:
            print(
                f"Warning: model '{args.model}' not found in Ollama. Available: {available_models}\n"
                f"You can pull it with: ollama pull {args.model}",
                file=sys.stderr,
            )
    except urllib.error.URLError:
        raise SystemExit(
            f"Could not connect to Ollama at {args.ollama_host}\n"
            "Make sure Ollama is running: ollama serve"
        )

    for sample_index in range(args.samples):
        theme = rng.choice(THEMES)
        last_error = None

        for attempt in range(1, args.attempts + 1):
            prompt = build_prompt(theme, used_room_names)
            raw_text = call_model(prompt, args)
            try:
                room = validate_room(extract_json_object(raw_text))
                used_room_names.append(room["room_name"])
                pending_rows.append(
                    {
                        "instruction": json.dumps(INSTRUCTION, separators=(",", ":")),
                        "response": json.dumps(room, ensure_ascii=False, separators=(",", ":")),
                    }
                )
                print(f"[{sample_index + 1}/{args.samples}] {room['room_name']}", flush=True)
                if args.format == "jsonl" and len(pending_rows) >= args.batch_size:
                    written_count += flush_batch(output_path, pending_rows, args.format)
                    print(f"Saved {written_count} rows to {args.output}", flush=True)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                print(
                    f"[{sample_index + 1}/{args.samples}] retry {attempt}/{args.attempts}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            skipped_count += 1
            print(
                f"[{sample_index + 1}/{args.samples}] skipped after {args.attempts} retries: {last_error}",
                file=sys.stderr,
                flush=True,
            )

    if pending_rows:
        if args.format == "jsonl":
            written_count += flush_batch(output_path, pending_rows, args.format)
        else:
            write_dataset(output_path, pending_rows, args.format)
            written_count += len(pending_rows)

    if skipped_count:
        print(f"Skipped {skipped_count} samples after retry exhaustion", file=sys.stderr, flush=True)

    return written_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate valid JSON room dataset rows using a local Ollama model."
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:4b",
        help="Ollama model name (e.g. qwen2.5:7b, llama3:8b). Must be pulled first.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API base URL.",
    )
    parser.add_argument("--output", default="F:/Projects/datasets/generate-room-dataset.jsonl", help="Output dataset path.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of valid samples to write.")
    parser.add_argument("--format", choices=["jsonl", "array"], default="jsonl")
    parser.add_argument("--batch-size", type=int, default=25, help="Save every N valid JSONL rows.")
    parser.add_argument("--attempts", type=int, default=5, help="Retries per sample.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repeat-penalty", type=float, default=1.08)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument(
        "--no-json-mode",
        dest="json_mode",
        action="store_false",
        help="Disable Ollama JSON mode (format=json).",
    )
    parser.set_defaults(json_mode=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_count = generate_dataset(args)
    print(f"Wrote {written_count} rows to {args.output}")


if __name__ == "__main__":
    main()
