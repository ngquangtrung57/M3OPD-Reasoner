"""Convert the Hugging Face release (M3OPD-SFT-1.35M, M3OPD-RL-193K) into trainer input files.

    python -m m3opd.data.prepare --sft-dir <local M3OPD-SFT-1.35M> --rl-dir <local M3OPD-RL-193K> --out <out dir>

Either directory is optional. Steps:

1. Unzip the media shards `<domain>/media_zips/*.zip` at the dataset root (they hold `<domain>/media/...`).
   A shard that was unzipped completely gets a marker file; an existing file of the right size is never rewritten.
2. SFT: write `<out>/sft/<domain>/part_NN.jsonl` in the lmms-engine chat format (media as absolute paths), and
   `<out>/sft/mix/<domain>_part_NN.jsonl`, the same rows in one flat schema (every content item carries the keys
   type, text, image_url.url and video_url.url, empty when unused; meta holds strings only). The flat schema lets
   lmms-engine concatenate the shards of several domains. Each shard stays below `--shard-bytes`.
3. RL: write `<out>/rl/<domain>/data.parquet` in the verl schema (media as absolute paths).

Domains of the release: image_stem, image_perception, video. Inside the training files they have the codes
imgr, imgp, vid: the code is `data_source`, `extra_info.domain` and `extra_info.teacher_key`, so a multi-teacher
trainer can route each prompt to the teacher of its domain.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

DOMAINS = {"image_stem": "imgr", "image_perception": "imgp", "video": "vid"}
SHARD_BYTES = 1_400_000_000          # below 1.5 GB per jsonl shard
RL_ROW_GROUP = 1024


# --------------------------------------------------------------------------- media
def _read_index(root: Path, domain: str) -> dict[str, str]:
    """media path -> shard file name, from `<domain>/media_index.tsv` ({} when the file is missing)."""
    f = root / domain / "media_index.tsv"
    out: dict[str, str] = {}
    if f.exists():
        with open(f) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    out[parts[0]] = parts[2]
    return out


def _safe_dest(root: Path, name: str) -> Path:
    dest = (root / name).resolve()
    if root.resolve() not in dest.parents:
        raise ValueError(f"unsafe path in zip shard: {name!r}")
    return dest


def extract_media(root: str | os.PathLike, domains: Iterable[str] = tuple(DOMAINS),
                  only: set[str] | None = None) -> int:
    """Unzip the media shards of `domains` below `root`. Returns the number of files written.

    Idempotent: a shard with a marker is skipped and an existing file of the right size is kept.
    `only` (a set of paths from the dataset root) limits the extraction to those files; no marker is written then.
    """
    root = Path(root)
    written = 0
    for dom in domains:
        shards = sorted(glob.glob(str(root / dom / "media_zips" / "*.zip")))
        if not shards:
            continue
        wanted: set[str] | None = None
        if only is not None:
            wanted = {p for p in only if p.startswith(f"{dom}/")}
            if not wanted:
                continue
            index = _read_index(root, dom)
            if index:
                names = {index[p] for p in wanted if p in index}
                shards = [s for s in shards if os.path.basename(s) in names]
        marker_dir = root / dom / ".extracted"
        for shard in shards:
            marker = marker_dir / os.path.basename(shard)
            if wanted is None and marker.exists():
                continue
            with zipfile.ZipFile(shard) as z:
                for info in z.infolist():
                    if info.is_dir() or (wanted is not None and info.filename not in wanted):
                        continue
                    dest = _safe_dest(root, info.filename)
                    if dest.exists() and dest.stat().st_size == info.file_size:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + ".tmp")
                    with z.open(info) as src, open(tmp, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
                    os.replace(tmp, dest)
                    written += 1
            if wanted is None:
                marker_dir.mkdir(parents=True, exist_ok=True)
                marker.touch()
    return written


def _abs(root: Path, rel: str) -> str:
    p = os.path.abspath(root / rel)
    if not os.path.exists(p):
        raise FileNotFoundError(f"media file {p} is missing; unzip the media shards of the dataset first")
    return p


# --------------------------------------------------------------------------- parquet input
def _iter_rows(dataset_dir: Path, domain: str, ids: set[str] | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(str(dataset_dir / domain / "data" / "*.parquet"))):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if ids is None or row["id"] in ids:
                    yield row


# --------------------------------------------------------------------------- SFT
def sft_row(row: dict, domain: str, root: Path) -> dict:
    """Release SFT row -> lmms-engine row (media slots filled in order from `images` and `videos`)."""
    code = DOMAINS[domain]
    images, videos = iter(row["images"] or []), iter(row["videos"] or [])
    messages = []
    for m in row["messages"]:
        items = []
        for c in m["content"]:
            if c["type"] == "text":
                items.append({"type": "text", "text": c["text"]})
            elif c["type"] == "image":
                items.append({"type": "image_url", "image_url": {"url": _abs(root, next(images))}})
            elif c["type"] == "video":
                items.append({"type": "video_url", "video_url": {"url": _abs(root, next(videos))}})
            else:
                raise ValueError(f"{row['id']}: unknown content type {c['type']!r}")
        messages.append({"role": m["role"], "content": items})
    source = row["source"]
    meta = {"domain": "sft0" if source.startswith("sft0_") else code, "card_id": source, "split": "sft",
            "task_type": row["task_type"], "pool_domain": code, "thinking": True}
    return {"id": row["id"], "messages": messages, "meta": meta}


def flatten_sft_row(r: dict) -> dict:
    """lmms-engine row -> flat schema (one struct for every content item, strings only in meta)."""
    messages = []
    for m in r["messages"]:
        items = []
        for c in m["content"]:
            items.append({"type": c["type"], "text": c.get("text", ""),
                          "image_url": {"url": c.get("image_url", {}).get("url", "")},
                          "video_url": {"url": c.get("video_url", {}).get("url", "")}})
        messages.append({"role": m["role"], "content": items})
    meta = {k: r["meta"][k] for k in ("domain", "card_id", "task_type", "pool_domain")}
    return {"id": r["id"], "messages": messages, "meta": {k: str(v) for k, v in meta.items()}}


class ShardWriter:
    """Writes jsonl lines to `<dir>/<prefix>part_NN.jsonl`; opens a new shard before a shard passes `limit` bytes."""

    def __init__(self, out_dir: Path, prefix: str, limit: int):
        self.dir, self.prefix, self.limit = out_dir, prefix, limit
        self.n, self.size, self.fh, self.files = 0, 0, None, []
        out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, obj: dict) -> None:
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
        if self.fh is None or (self.size and self.size + len(data) > self.limit):
            self.close()
            path = self.dir / f"{self.prefix}part_{self.n:02d}.jsonl"
            self.fh, self.size, self.n = open(path, "wb"), 0, self.n + 1
            self.files.append(path)
        self.fh.write(data)
        self.size += len(data)

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def convert_sft(sft_dir: str | os.PathLike, out: str | os.PathLike, domains: Iterable[str] = tuple(DOMAINS),
                shard_bytes: int = SHARD_BYTES, ids: set[str] | None = None) -> dict[str, int]:
    """Write the SFT shards of every domain that exists in `sft_dir`. Returns rows per domain."""
    root, out = Path(sft_dir).resolve(), Path(out)
    counts: dict[str, int] = {}
    for dom in domains:
        if not (root / dom / "data").is_dir():
            continue
        plain = ShardWriter(out / "sft" / dom, "", shard_bytes)
        flat = ShardWriter(out / "sft" / "mix", f"{dom}_", shard_bytes)
        n = 0
        for row in _iter_rows(root, dom, ids):
            r = sft_row(row, dom, root)
            plain.write(r)
            flat.write(flatten_sft_row(r))
            n += 1
        plain.close()
        flat.close()
        counts[dom] = n
    return counts


# --------------------------------------------------------------------------- RL
def rl_schema():
    import pyarrow as pa
    return pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("images", pa.list_(pa.struct([("image", pa.string()), ("bytes", pa.binary())]))),
        ("videos", pa.list_(pa.struct([("video", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        ("extra_info", pa.struct([("id", pa.string()), ("domain", pa.string()), ("card_id", pa.string()),
                                  ("split", pa.string()), ("task_type", pa.string()), ("teacher_key", pa.string()),
                                  ("thinking", pa.bool_()), ("tolerance", pa.float64())])),
    ])


def rl_row(row: dict, domain: str, root: Path) -> dict:
    """Release RL row -> verl row (media as absolute paths)."""
    code = DOMAINS[domain]
    tol = row.get("tolerance")
    return {
        "data_source": code,
        "prompt": [{"role": m["role"], "content": m["content"]} for m in row["prompt"]],
        "images": [{"image": _abs(root, p), "bytes": None} for p in row["images"] or []],
        "videos": [{"video": _abs(root, p)} for p in row["videos"] or []],
        "ability": row["task_type"],
        "reward_model": {"style": "rule", "ground_truth": row["ground_truth"] if row["ground_truth"] is not None else ""},
        "extra_info": {"id": row["id"], "domain": code, "card_id": row["source"], "split": "rl",
                       "task_type": row["task_type"], "teacher_key": code, "thinking": True,
                       "tolerance": float(tol) if tol is not None else None},
    }


def convert_rl(rl_dir: str | os.PathLike, out: str | os.PathLike, domains: Iterable[str] = tuple(DOMAINS),
               ids: set[str] | None = None) -> dict[str, int]:
    """Write `<out>/rl/<domain>/data.parquet` for every domain that exists in `rl_dir`. Returns rows per domain."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, out = Path(rl_dir).resolve(), Path(out)
    counts: dict[str, int] = {}
    for dom in domains:
        if not (root / dom / "data").is_dir():
            continue
        d = out / "rl" / dom
        d.mkdir(parents=True, exist_ok=True)
        schema, n, buf = rl_schema(), 0, []
        tmp = d / "data.parquet.tmp"
        with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
            for row in _iter_rows(root, dom, ids):
                buf.append(rl_row(row, dom, root))
                n += 1
                if len(buf) >= RL_ROW_GROUP:
                    w.write_table(pa.Table.from_pylist(buf, schema=schema))
                    buf = []
            if buf:
                w.write_table(pa.Table.from_pylist(buf, schema=schema))
        os.replace(tmp, d / "data.parquet")
        counts[dom] = n
    return counts


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sft-dir", help="local copy of M3OPD-SFT-1.35M")
    ap.add_argument("--rl-dir", help="local copy of M3OPD-RL-193K")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--domains", nargs="+", choices=list(DOMAINS), default=list(DOMAINS))
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES, help="size limit of one SFT jsonl shard")
    ap.add_argument("--skip-extract", action="store_true", help="do not unzip the media shards")
    a = ap.parse_args(argv)
    if not (a.sft_dir or a.rl_dir):
        ap.error("give --sft-dir, --rl-dir or both")
    for name, d in (("sft", a.sft_dir), ("rl", a.rl_dir)):
        if not d:
            continue
        if not a.skip_extract:
            print(f"{name}: unzip media", flush=True)
            print(f"{name}: {extract_media(d, a.domains)} media files written", flush=True)
        conv = convert_sft(d, a.out, a.domains, a.shard_bytes) if name == "sft" else convert_rl(d, a.out, a.domains)
        for dom, n in conv.items():
            print(f"{name}/{dom}: {n} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
