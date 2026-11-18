"""Finish a merged Hugging Face checkpoint so that transformers and vLLM load it like Qwen3.5-9B.

    python -m m3opd.ckpt <merged_hf_dir> --base <Qwen3.5-9B dir>

1. Both mergers (lmms-engine and verl) write the vision tower as `model.language_model.visual.*`; vLLM expects
   `model.visual.*`. Only the safetensors headers change (the tensor bytes are copied).
2. generation_config.json gets both end tokens (<|im_end|>, <|endoftext|>) as eos_token_id.
3. The tokenizer and processor files are copied from the base model (the mergers re-serialize them and store the
   training-time pixel limits).
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

EOS_IDS = [248046, 248044]  # <|im_end|>, <|endoftext|>
PROCESSOR_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
                   "video_preprocessor_config.json", "processor_config.json")
BAD_VISUAL = "model.language_model.visual."
GOOD_VISUAL = "model.visual."


def fix_visual_keys(ckpt_dir: Path) -> int:
    """Rename `model.language_model.visual.*` to `model.visual.*` in every *.safetensors; return the count."""
    n_fixed = 0
    for f in sorted(Path(ckpt_dir).glob("*.safetensors")):
        with open(f, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        bad = [k for k in header if k.startswith(BAD_VISUAL)]
        if not bad:
            continue
        new = {(GOOD_VISUAL + k[len(BAD_VISUAL):] if k.startswith(BAD_VISUAL) else k): v for k, v in header.items()}
        if len(new) != len(header):
            raise ValueError(f"{f}: renaming visual keys collides with existing keys")
        blob = json.dumps(new, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)   # keep the data block 8-byte aligned
        tmp = f.with_name(f.name + ".tmp")
        with open(f, "rb") as src, open(tmp, "wb") as dst:
            src.seek(8 + hlen)
            dst.write(struct.pack("<Q", len(blob)) + blob)
            while chunk := src.read(64 << 20):
                dst.write(chunk)
        tmp.replace(f)
        n_fixed += len(bad)
    index = Path(ckpt_dir) / "model.safetensors.index.json"
    if n_fixed and index.is_file():
        idx = json.loads(index.read_text())
        idx["weight_map"] = {(GOOD_VISUAL + k[len(BAD_VISUAL):] if k.startswith(BAD_VISUAL) else k): v
                             for k, v in idx["weight_map"].items()}
        index.write_text(json.dumps(idx, indent=2) + "\n")
    return n_fixed


def write_generation_config(ckpt_dir: Path, base_dir: Path) -> None:
    out = Path(ckpt_dir) / "generation_config.json"
    cfg = {}
    for src in (out, Path(base_dir) / "generation_config.json"):
        if src.is_file():
            cfg = json.loads(src.read_text())
            break
    cfg["eos_token_id"] = list(EOS_IDS)
    cfg.setdefault("do_sample", True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")


def copy_processor_files(dst: Path, base_dir: Path) -> list[str]:
    dst, base_dir = Path(dst), Path(base_dir)
    copied = []
    for f in PROCESSOR_FILES:
        src, out = base_dir / f, dst / f
        if src.is_file():
            shutil.copyfile(src, out)
            copied.append(f)
        elif out.is_file():
            out.unlink()
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir")
    ap.add_argument("--base", required=True, help="Qwen3.5-9B model dir (tokenizer, processor, generation config)")
    a = ap.parse_args()
    print("visual keys renamed:", fix_visual_keys(Path(a.ckpt_dir)))
    write_generation_config(Path(a.ckpt_dir), Path(a.base))
    print("processor files copied:", copy_processor_files(Path(a.ckpt_dir), Path(a.base)))


if __name__ == "__main__":
    main()
