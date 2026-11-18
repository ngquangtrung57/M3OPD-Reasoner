"""Render an SFT config of configs/sft/ into a plain lmms-engine config.

    python -m m3opd.sft_config configs/sft/sft_image_stem.yaml out.yaml

${VAR} values come from the environment (DATA_DIR, BASE_MODEL, OUTPUT_DIR). `dataset_config.datasets_glob` becomes
the inline `datasets` list that the qwen3_vl_iterable dataset expects (one jsonl file per entry, absolute media paths,
so `data_folder` is empty).
"""
from __future__ import annotations

import glob
import os
import re
import sys

import yaml

_VAR = re.compile(r"\$\{(\w+)\}")


def _expand(v):
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, str):
        def sub(m):
            if m.group(1) not in os.environ:
                raise KeyError(f"environment variable {m.group(1)} is not set")
            return os.environ[m.group(1)]
        return _VAR.sub(sub, v)
    return v


def render(path: str) -> dict:
    cfg = _expand(yaml.safe_load(open(path)))
    d = cfg["dataset_config"]
    files = sorted(glob.glob(d.pop("datasets_glob")))
    if not files:
        raise FileNotFoundError("no SFT shards match datasets_glob; run python -m m3opd.data.prepare first")
    d["datasets"] = [{"path": f, "data_type": "jsonl", "data_folder": ""} for f in files]
    return cfg


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    cfg = render(src)
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"{dst}: {len(cfg['dataset_config']['datasets'])} shards")


if __name__ == "__main__":
    main()
