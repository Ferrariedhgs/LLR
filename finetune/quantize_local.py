"""
quantize_local.py — quantise a fp16 GGUF to Q8_0 and Q4_K_M using a local
llama.cpp installation.

Usage:
    python quantize_local.py
    python quantize_local.py --input ./output/nemotron-room-lora-fp16.gguf
    python quantize_local.py --input model.gguf --llama-cpp C:/llama.cpp/build/bin

Requirements:
    llama.cpp built locally. If you don't have it yet:

    Windows (CUDA):
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp
        cmake -B build -DGGML_CUDA=ON
        cmake --build build --config Release

    The llama-quantize binary will be at:
        build/bin/Release/llama-quantize.exe  (Windows)
        build/bin/llama-quantize              (Linux/Mac)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Config — edit these defaults if needed
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = "F:/Projects/AI/finetunes/nemotron-nano-3-4b-escaper/nemotron-room-lora-fp16.gguf"

# Common llama.cpp build locations — the script tries all of them
LLAMA_CPP_SEARCH_PATHS = [
    "F:/Projects/AI/llama-cpp/llama.cpp/build/bin/Release",
    # Windows cmake Release build
    "./llama.cpp/build/bin/Release",
    "../llama.cpp/build/bin/Release",
    "C:/llama.cpp/build/bin/Release",
    # Windows cmake Debug build
    "./llama.cpp/build/bin/Debug",
    # Linux / Mac cmake build
    "./llama.cpp/build/bin",
    "../llama.cpp/build/bin",
    "/opt/llama.cpp/build/bin",
    # Pre-built binary in PATH
    "",
]

QUANTIZE_BIN_NAMES = ["llama-quantize.exe", "llama-quantize"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_quantize_binary(hint: str | None) -> Path:
    """Find the llama-quantize binary, optionally starting from a hint path."""
    search = ([hint] if hint else []) + LLAMA_CPP_SEARCH_PATHS
    for directory in search:
        for name in QUANTIZE_BIN_NAMES:
            candidate = Path(directory) / name if directory else Path(name)
            if candidate.exists():
                return candidate.resolve()
            # Also try via PATH (empty directory = just the binary name)
            if not directory:
                result = subprocess.run(
                    ["where" if sys.platform == "win32" else "which", name],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return Path(result.stdout.strip().splitlines()[0])
    raise FileNotFoundError(
        "Could not find llama-quantize binary.\n"
        "Build llama.cpp first:\n"
        "  git clone https://github.com/ggerganov/llama.cpp\n"
        "  cd llama.cpp\n"
        "  cmake -B build -DGGML_CUDA=ON\n"
        "  cmake --build build --config Release\n"
        "Then pass --llama-cpp ./llama.cpp/build/bin/Release"
    )


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",   # llama-quantize prints non-UTF-8 progress bars
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quantise a fp16 GGUF to Q8_0 and Q4_K_M using llama.cpp"
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to the fp16 GGUF file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to write quantised files (default: same dir as input)",
    )
    parser.add_argument(
        "--llama-cpp",
        default=None,
        help="Path to llama.cpp bin directory containing llama-quantize",
    )
    parser.add_argument(
        "--quants",
        nargs="+",
        default=["Q8_0", "Q4_K_M"],
        choices=["Q4_0", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"],
        help="Quantisation types to produce (default: Q8_0 Q4_K_M)",
    )
    args = parser.parse_args()

    # Resolve paths
    fp16_path = Path(args.input).resolve()
    if not fp16_path.exists():
        sys.exit(f"ERROR: Input file not found: {fp16_path}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else fp16_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find binary
    try:
        quantize_bin = find_quantize_binary(args.llama_cpp)
        print(f"Found llama-quantize: {quantize_bin}")
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")

    # Derive output filename stem from input
    stem = fp16_path.stem  # e.g. "nemotron-room-lora-fp16"
    # Strip trailing -fp16 / _fp16 suffix if present so output names are clean
    clean_stem = stem.removesuffix("-fp16").removesuffix("_fp16")

    # Quantise
    results = {}
    for quant in args.quants:
        out_name = f"{clean_stem}-{quant.lower().replace('_', '-')}.gguf"
        out_path = out_dir / out_name
        try:
            run(
                [str(quantize_bin), str(fp16_path), str(out_path), quant],
                label=f"Quantising → {quant}",
            )
            size_mb = out_path.stat().st_size / 1024 / 1024
            results[quant] = (out_path, size_mb, True)
            print(f"  ✓ {out_name}  ({size_mb:.0f} MB)")
        except RuntimeError as e:
            results[quant] = (out_path, 0, False)
            print(f"  ✗ {quant} failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    fp16_mb = fp16_path.stat().st_size / 1024 / 1024
    print(f"  fp16 (source):  {fp16_path.name}  ({fp16_mb:.0f} MB)")
    for quant, (path, size_mb, ok) in results.items():
        status = "✓" if ok else "✗ FAILED"
        print(f"  {quant:<10} {status}  {path.name}  ({size_mb:.0f} MB)")

    if all(ok for _, _, ok in results.values()):
        print("\nDone. All quants complete.")
    else:
        sys.exit("\nDone with errors — some quants failed.")


if __name__ == "__main__":
    main()
