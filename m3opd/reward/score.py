"""Rule reward. Default config: configs/reward.yaml in this package.

verl entry: custom_reward_function {path: this file, name: compute_score}.

R = w_acc*acc + w_fmt*fmt (0.9 / 0.1 in the reward config); acc = 0 unless fmt == 1.
Format (think mode, native Qwen3.5 template puts '<think>\\n' in the prompt): the response is
'reasoning </think> answer' with exactly one '</think>' and no other think tag; no-think mode: no think
tags at all. Boxed task types also need exactly one \\boxed{} in the answer part (else fmt = 0.5).
The task_type comes from extra_info.task_type; thinking from extra_info.thinking or the prompt config table.
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):  # loaded by file path (verl custom_reward_function): make 'm3opd' importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3opd import prompts  # noqa: E402
from m3opd.reward import types as T  # noqa: E402

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "reward.yaml"
_TRAILING_SPECIAL = ("<|im_end|>", "<|endoftext|>")
_warned: set[str] = set()


@functools.cache
def load_cfg(path: str | None = None) -> dict[str, Any]:
    """Load the reward config. Env M3OPD_REWARD_CFG or reward_kwargs 'reward_cfg' override the path."""
    return yaml.safe_load(Path(path or os.environ.get("M3OPD_REWARD_CFG") or DEFAULT_CFG).read_text())


def _clean(response: str) -> str:
    s = response.rstrip()
    changed = True
    while changed:
        changed = False
        for tok in _TRAILING_SPECIAL:
            if s.endswith(tok):
                s, changed = s[: -len(tok)].rstrip(), True
    return s


def split_response(response: str, thinking: bool, cfg: dict) -> str | None:
    """Answer part of a well-formed response, or None when the think structure is wrong."""
    f = cfg.get("format", {})
    s = _clean(response)
    if thinking:
        if f.get("allow_leading_think_tag", True) and s.lstrip().startswith("<think>"):
            s = s.lstrip()[len("<think>"):]
        if s.count("</think>") != 1 or "<think>" in s:
            return None
        reasoning, answer = s.split("</think>")
        if f.get("require_nonempty_think", True) and not reasoning.strip():
            return None
    else:
        if "<think>" in s or "</think>" in s:
            return None
        answer = s
    return answer.strip() or None


def format_score(response: str, task_type: str, thinking: bool, cfg: dict) -> tuple[float, str | None]:
    """(fmt, answer payload). Payload = the single \\boxed{} content for boxed types, else the answer."""
    answer = split_response(response, thinking, cfg)
    if answer is None:
        return 0.0, None
    f = cfg.get("format", {})
    if task_type not in f.get("boxed_task_types", list(T.BOXED_SCORERS)):
        return 1.0, answer
    boxes = T.extract_boxed(answer)
    if len(boxes) == 1 and boxes[0] and answer.count("\\boxed") == 1:
        return 1.0, boxes[0]
    return float(f.get("partial_box_score", 0.5)), None


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None,
                  **kw) -> dict:
    """verl custom reward. Returns {score, acc, fmt, task_type}."""
    cfg = load_cfg(kw.get("reward_cfg"))
    info = dict(extra_info or {})
    task_type = str(info.get("task_type") or kw.get("task_type") or "")
    if task_type not in prompts.TASK_TYPES:
        if task_type not in _warned:
            _warned.add(task_type)
            print(f"[m3opd.reward] unknown task_type {task_type!r} (data_source={data_source}); score 0",
                  file=sys.stderr)
        return {"score": 0.0, "acc": 0.0, "fmt": 0.0, "task_type": task_type or "unknown"}
    thinking = info.get("thinking")
    if not isinstance(thinking, bool):
        thinking = prompts.thinking_for(info.get("domain"), task_type, info.get("meta"))

    fmt, payload = format_score(str(solution_str), task_type, thinking, cfg)
    acc = 0.0
    full = fmt >= 1.0 or not cfg.get("format", {}).get("acc_requires_full_fmt", True)
    if full and payload is not None:
        if task_type in ("math", "short", "count", "mc"):  # bound sympy work; OCR text stays whole
            cap = int(cfg.get("limits", {}).get("max_answer_chars", 300))
            payload = payload[: max(cap, 2 * len(str(ground_truth)))]   # a long gold keeps a long answer whole
        try:
            if task_type == "if":
                acc = T.score_if(payload, ground_truth, cfg)
            elif task_type in T.NEEDS_EXTRA:
                acc = T.BOXED_SCORERS[task_type](payload, ground_truth, cfg, info)
            elif task_type in T.BOXED_SCORERS:
                acc = T.BOXED_SCORERS[task_type](payload, ground_truth, cfg)
        except Exception as e:  # a scorer bug must not kill a training step
            print(f"[m3opd.reward] {task_type} scorer error: {type(e).__name__}: {e}", file=sys.stderr)
            acc = 0.0
    w = cfg.get("weights", {"acc": 0.9, "fmt": 0.1})
    acc = float(acc)
    return {"score": float(w["acc"]) * acc + float(w["fmt"]) * fmt, "acc": acc, "fmt": float(fmt),
            "task_type": task_type}
