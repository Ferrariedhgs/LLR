"""
Modal LoRA fine-tuning for unsloth/Qwen3.5-4B
on a folder of JSONL datasets, then exports to GGUF in fp16, q8_0, q4_K_M.

Usage:
    modal run qwen3_lora.py                          # full run
    modal run qwen3_lora.py --data-dir ./my_data     # custom data folder
    modal run qwen3_lora.py --dry-run                # validate dataset only

Dataset JSONL format (any of these row shapes are handled):
  {"instruction": "...", "response": "..."}
  {"container": "...", "description": "..."}
  {"key_name": "...", "description": "..."}
  {"room_name": "...", "description": "..."}
  etc. — rows are normalised to instruction/response pairs before training.

Outputs written to ./output/ (local) after the run:
  qwen3-room-lora-fp16.gguf
  qwen3-room-lora-q8_0.gguf
  qwen3-room-lora-q4_K_M.gguf
"""

import json
import os
import re
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image — everything needed for unsloth + llama.cpp GGUF export
# ---------------------------------------------------------------------------

BASE_MODEL = "unsloth/Qwen3.5-4B"
VOLUME_NAME = "qwen3-lora-vol"

# Persistent volume to cache the base model weights between runs
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Qwen3.5 is a standard transformer — no custom kernels needed.
# We use the lighter cudnn-runtime base (no nvcc required) and skip
# causal-conv1d / mamba-ssm entirely.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git", "curl", "wget", "cmake", "build-essential",
        "libgomp1", "ninja-build",
    )
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
    )
    # Step 1: numpy + packaging before torch
    .pip_install("numpy", "packaging")
    # Step 2: torch cu124
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Step 3: training stack
    .pip_install(
        "trl>=0.8.6",
        "huggingface_hub",
        "datasets",
        "sentencepiece",
        "accelerate",
        "transformers>=4.46.0",
    )
    # Step 4: unsloth — sees torch already satisfied, won't upgrade it
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )
    # Step 5: llama.cpp — CPU-only build (no GPU at image-build time)
    .run_commands(
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=OFF"
        " && cmake --build build --config Release -j$(nproc)",
        "pip install -r /opt/llama.cpp/requirements.txt",
    )
)

app = modal.App("qwen3-lora-train-v1", image=image)

# ---------------------------------------------------------------------------
# Dataset helpers  (runs locally to build the dataset, uploaded to the volume)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a creative dungeon-master AI that generates escape-room content. "
    "Always respond with a single valid JSON object and nothing else."
)


def _row_to_instruction_response(row: dict) -> dict | None:
    """
    Normalise any of the observed JSONL row shapes into
    {"instruction": str, "response": str}.

    Rows that already have instruction/response are passed through.
    Partial rows (container, key_name, room_name …) are reconstructed
    into a minimal generate_room task/response pair.
    """
    # Already a clean instruction/response pair
    if "instruction" in row and "response" in row:
        try:
            instr = row["instruction"]
            if isinstance(instr, str):
                instr = json.loads(instr)
            resp = row["response"]
            if isinstance(resp, str):
                resp = json.loads(resp)
            return {
                "instruction": json.dumps(instr, ensure_ascii=False),
                "response": json.dumps(resp, ensure_ascii=False),
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "instruction": str(row["instruction"]),
                "response": str(row["response"]),
            }

    # Partial rows — reconstruct a plausible task/response
    if "room_name" in row:
        task = {"task": "generate_room"}
        resp = {"room_name": row["room_name"], "room_story": row.get("description", "")}
        return {
            "instruction": json.dumps(task),
            "response": json.dumps(resp, ensure_ascii=False),
        }

    if "container" in row:
        task = {"task": "generate_container"}
        resp = {"container_name": row["container"], "container_prompt": row.get("description", "")}
        return {
            "instruction": json.dumps(task),
            "response": json.dumps(resp, ensure_ascii=False),
        }

    if "key_name" in row:
        task = {"task": "generate_key"}
        resp = {"key_name": row["key_name"], "key_prompt": row.get("description", "")}
        return {
            "instruction": json.dumps(task),
            "response": json.dumps(resp, ensure_ascii=False),
        }

    # Unknown shape — skip
    return None


def load_jsonl_folder(folder: str) -> list[dict]:
    """Load and normalise all .jsonl files in a folder."""
    records = []
    folder_path = Path(folder)
    files = list(folder_path.glob("**/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {folder}")
    for path in files:
        print(f"  Loading {path} …")
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"    [warn] {path}:{lineno} – skipping bad JSON: {e}")
                    continue
                normalised = _row_to_instruction_response(row)
                if normalised:
                    records.append(normalised)
                else:
                    print(f"    [warn] {path}:{lineno} – unrecognised row shape, skipping")
    print(f"  Total usable rows: {len(records)}")
    return records


def format_chat_prompt(record: dict, tokenizer) -> str:
    """Format as a Qwen3 chat template (system + user + assistant)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": record["instruction"]},
        {"role": "assistant", "content": record["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",          # swap to H100 for speed
    timeout=60 * 60 * 3,                           # 3h hard limit
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN
)
def train(
    dataset_records: list[dict],
    hf_model_id: str = BASE_MODEL,
    output_dir: str = "/vol/lora-output",
    # LoRA hyper-params
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    # Training hyper-params
    max_seq_length: int = 2048,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    logging_steps: int = 10,
) -> str:
    """Fine-tune with unsloth LoRA, merge, export three GGUF quant levels."""
    import subprocess, torch

    # ── Runtime diagnostics ────────────────────────────────────────────────
    print("=== TORCH DIAGNOSTICS ===")
    print(f"  torch version    : {torch.__version__}")
    print(f"  cuda available   : {torch.cuda.is_available()}")
    print(f"  cuda version     : {torch.version.cuda}")
    print(f"  device count     : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  device name      : {torch.cuda.get_device_name(0)}")
    r = subprocess.run("pip show torch | grep -E 'Name|Version|Location'",
                       shell=True, capture_output=True, text=True)
    print(f"  pip torch info   :\n{r.stdout.strip()}")
    print("=========================")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False on this container. "
            f"torch={torch.__version__}, cuda={torch.version.cuda}. "
            "Check the image build logs — torch may be a CPU wheel."
        )

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base model with unsloth 4-bit quantisation for training
    # ------------------------------------------------------------------
    print("=== Loading base model ===")
    model_cache = "/vol/model-cache"
    os.makedirs(model_cache, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_id,
        max_seq_length=max_seq_length,
        dtype=None,           # auto-detect (bf16 on A100)
        load_in_4bit=True,    # QLoRA
        cache_dir=model_cache,
        token=os.environ.get("HF_TOKEN"),
    )

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    print("=== Attaching LoRA adapters ===")
    for name, _ in model.named_modules():
        print(name)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
    )

    # ------------------------------------------------------------------
    # 3. Prepare dataset
    # ------------------------------------------------------------------
    print("=== Preparing dataset ===")

    def _format(batch):
        return {
            "text": [
                format_chat_prompt({"instruction": instr, "response": resp}, tokenizer)
                for instr, resp in zip(batch["instruction"], batch["response"])
            ]
        }

    hf_dataset = Dataset.from_list(dataset_records)
    hf_dataset = hf_dataset.map(_format, batched=True, remove_columns=["instruction", "response"])
    print(f"  Dataset size: {len(hf_dataset)} examples")
    print(f"  Sample prompt (first 300 chars):\n  {hf_dataset[0]['text'][:300]}")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("=== Training ===")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=42,
            report_to="none",
        ),
    )
    trainer.train()
    print("=== Training complete ===")

    # ------------------------------------------------------------------
    # 5. Merge LoRA into full fp16 weights and save
    # ------------------------------------------------------------------
    print("=== Merging LoRA → fp16 ===")
    merged_dir = os.path.join(output_dir, "merged-fp16")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # ------------------------------------------------------------------
    # 6. Convert to GGUF + quantise
    # ------------------------------------------------------------------
    print("=== Converting to GGUF ===")
    convert_py = "/opt/llama.cpp/convert_hf_to_gguf.py"
    quantize_bin = "/opt/llama.cpp/build/bin/llama-quantize"

    gguf_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    fp16_gguf  = os.path.join(gguf_dir, "qwen3-room-lora-fp16.gguf")
    q8_gguf    = os.path.join(gguf_dir, "qwen3-room-lora-q8_0.gguf")
    q4km_gguf  = os.path.join(gguf_dir, "qwen3-room-lora-q4_K_M.gguf")

    # fp16 GGUF (base conversion)
    _run(f"python {convert_py} {merged_dir} --outfile {fp16_gguf} --outtype f16")

    # q8_0
    _run(f"{quantize_bin} {fp16_gguf} {q8_gguf} Q8_0")

    # q4_K_M
    _run(f"{quantize_bin} {fp16_gguf} {q4km_gguf} Q4_K_M")

    volume.commit()
    print(f"=== All done. GGUFs in {gguf_dir} ===")
    return gguf_dir


def _run(cmd: str):
    """Run a shell command, raise on failure."""
    import subprocess
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-2000:])   # tail to avoid flooding logs
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{result.stderr[-2000:]}")




@app.function(gpu="A100")
def debug_env():
    """Standalone diagnostic — run with: modal run qwen3_lora.py::debug_env"""
    import subprocess, torch
    print(f"torch version  : {torch.__version__}")
    print(f"cuda available : {torch.cuda.is_available()}")
    print(f"cuda version   : {torch.version.cuda}")
    print(f"device count   : {torch.cuda.device_count()}")
    r = subprocess.run("pip show torch", shell=True, capture_output=True, text=True)
    print(r.stdout)
    r2 = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    print(r2.stdout or r2.stderr)


# ---------------------------------------------------------------------------
# Download helper — pulls GGUFs from the Modal volume to local disk
# ---------------------------------------------------------------------------

@app.function(volumes={"/vol": volume})
def list_outputs() -> list[str]:
    paths = []
    for root, _, files in os.walk("/vol/lora-output/gguf"):
        for f in files:
            paths.append(os.path.join(root, f))
    return paths


@app.function(volumes={"/vol": volume})
def read_file(remote_path: str) -> bytes:
    with open(remote_path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    data_dir: str = "./data",
    dry_run: bool = False,
    output_local: str = "./output",
):
    """
    Parameters
    ----------
    data_dir      : local folder with .jsonl files (default: ./data)
    dry_run       : only validate & print dataset stats, skip training
    output_local  : local directory to download GGUFs into (default: ./output)
    """
    print(f"Loading dataset from: {data_dir}")
    records = load_jsonl_folder(data_dir)

    if dry_run:
        print("\n--- DRY RUN: first 3 normalised records ---")
        for r in records[:3]:
            print(json.dumps(r, indent=2, ensure_ascii=False))
        print("\nDataset OK — skipping training (--dry-run).")
        return

    print(f"\nStarting remote training with {len(records)} examples …")
    gguf_dir = train.remote(records)

    # Download results
    print(f"\nDownloading GGUFs from {gguf_dir} → {output_local}")
    os.makedirs(output_local, exist_ok=True)
    remote_files = list_outputs.remote()
    for remote_path in remote_files:
        fname = os.path.basename(remote_path)
        local_path = os.path.join(output_local, fname)
        print(f"  {fname} …", end=" ", flush=True)
        data = read_file.remote(remote_path)
        with open(local_path, "wb") as fh:
            fh.write(data)
        size_mb = len(data) / 1024 / 1024
        print(f"{size_mb:.1f} MB")

    print(f"\nAll files saved to {output_local}/")
    print("  qwen3-room-lora-fp16.gguf")
    print("  qwen3-room-lora-q8_0.gguf")
    print("  qwen3-room-lora-q4_K_M.gguf")
