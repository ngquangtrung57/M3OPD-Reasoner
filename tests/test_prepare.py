"""Round-trip test of m3opd.data.prepare on a tiny synthetic copy of the Hugging Face layout."""
import json
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3opd.data import prepare as P  # noqa: E402

MEDIA = {"image_stem": ("png", b"\x89PNG-stem"), "image_perception": ("jpg", b"\xff\xd8-perc"),
         "video": ("mp4", b"video-bytes" * 10)}
SFT_SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("task_type", pa.string()),
                        ("generator", pa.string()),
                        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.list_(
                            pa.struct([("type", pa.string()), ("text", pa.string())])))]))),
                        ("images", pa.list_(pa.string())), ("videos", pa.list_(pa.string()))])
RL_SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("task_type", pa.string()),
                       ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
                       ("images", pa.list_(pa.string())), ("videos", pa.list_(pa.string())),
                       ("ground_truth", pa.string()), ("tolerance", pa.float64())])


def media_path(dom):
    return f"{dom}/media/ab/file.{MEDIA[dom][0]}"


def build(root: Path, kind: str, n: int = 5):
    for dom in P.DOMAINS:
        ext, blob = MEDIA[dom]
        slot = "video" if dom == "video" else "image"
        rows = []
        for i in range(n):
            imgs, vids = ([], [media_path(dom)]) if dom == "video" else ([media_path(dom)], [])
            if kind == "sft":
                rows.append({"id": f"s_{dom}:{i}~g0", "source": "sft0_x" if i == 0 else f"s_{dom}", "task_type": "mc",
                             "generator": "g",
                             "messages": [{"role": "user", "content": [{"type": slot, "text": None},
                                                                       {"type": "text", "text": f"Q{i}"}]},
                                          {"role": "assistant", "content": [{"type": "text", "text": f"<think>\nt\n</think>\n\nA{i}"}]}],
                             "images": imgs, "videos": vids})
            else:
                rows.append({"id": f"r_{dom}:{i}", "source": f"r_{dom}", "task_type": "math",
                             "prompt": [{"role": "user", "content": f"<{slot}>Q{i}"}], "images": imgs, "videos": vids,
                             "ground_truth": str(i), "tolerance": 0.5 if i == 1 else None})
        d = root / dom
        (d / "data").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=SFT_SCHEMA if kind == "sft" else RL_SCHEMA),
                       d / "data" / "train-00000-of-00001.parquet")
        (d / "media_zips").mkdir()
        with zipfile.ZipFile(d / "media_zips" / f"{dom}-00000.zip", "w", zipfile.ZIP_STORED) as z:
            z.writestr(media_path(dom), blob)
        (d / "media_index.tsv").write_text(f"{media_path(dom)}\t{len(blob)}\t{dom}-00000.zip\n")


@pytest.fixture()
def tree(tmp_path):
    build(tmp_path / "sft", "sft")
    build(tmp_path / "rl", "rl")
    return tmp_path


def test_round_trip(tree):
    out = tree / "out"
    argv = ["--sft-dir", str(tree / "sft"), "--rl-dir", str(tree / "rl"), "--out", str(out)]
    assert P.main(argv) == 0
    for dom, code in P.DOMAINS.items():
        ext, blob = MEDIA[dom]
        mp = tree / "sft" / media_path(dom)
        assert mp.read_bytes() == blob
        lines = [json.loads(x) for x in (out / "sft" / dom / "part_00.jsonl").read_text().splitlines()]
        assert len(lines) == 5
        item = lines[1]["messages"][0]["content"][0]
        key = "video_url" if dom == "video" else "image_url"
        assert item == {"type": key, key: {"url": str(mp)}}
        assert lines[1]["messages"][0]["content"][1] == {"type": "text", "text": "Q1"}
        assert lines[1]["messages"][1]["content"][0]["text"].endswith("A1")
        assert lines[1]["meta"]["domain"] == code and lines[0]["meta"]["domain"] == "sft0"
        flat = [json.loads(x) for x in (out / "sft" / "mix" / f"{dom}_part_00.jsonl").read_text().splitlines()]
        c0 = flat[1]["messages"][0]["content"][0]
        assert set(c0) == {"type", "text", "image_url", "video_url"} and c0["text"] == ""
        assert c0[key]["url"] == str(mp)
        assert flat[1]["meta"] == {"domain": code, "card_id": f"s_{dom}", "task_type": "mc", "pool_domain": code}
        t = pq.read_table(out / "rl" / dom / "data.parquet").to_pylist()
        assert len(t) == 5
        r = t[1]
        assert r["data_source"] == code and r["ability"] == "math"
        assert r["reward_model"] == {"style": "rule", "ground_truth": "1"}
        assert r["extra_info"] == {"id": f"r_{dom}:1", "domain": code, "card_id": f"r_{dom}", "split": "rl",
                                   "task_type": "math", "teacher_key": code, "thinking": True, "tolerance": 0.5}
        assert t[0]["extra_info"]["tolerance"] is None
        rmp = str(tree / "rl" / media_path(dom))
        if dom == "video":
            assert r["videos"] == [{"video": rmp}] and r["images"] == []
        else:
            assert r["images"] == [{"image": rmp, "bytes": None}] and r["videos"] == []
        assert r["prompt"] == [{"role": "user", "content": f"<{'video' if dom == 'video' else 'image'}>Q1"}]


def test_extract_is_idempotent(tree):
    root = tree / "sft"
    assert P.extract_media(root) == 3
    assert P.extract_media(root) == 0
    f = root / media_path("video")
    f.unlink()
    (root / "video" / ".extracted" / "video-00000.zip").unlink()
    assert P.extract_media(root) == 1 and f.exists()


def test_sft_shards_stay_below_limit(tree):
    out = tree / "out"
    P.extract_media(tree / "sft")
    P.convert_sft(tree / "sft", out, shard_bytes=2000)
    parts = sorted((out / "sft" / "image_stem").glob("part_*.jsonl"))
    assert len(parts) > 1 and all(p.stat().st_size <= 2000 for p in parts)
    assert sum(len(p.read_text().splitlines()) for p in parts) == 5


def test_missing_media_is_an_error(tree):
    with pytest.raises(FileNotFoundError):
        P.convert_rl(tree / "rl", tree / "out")
