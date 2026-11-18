"""Prompt standard. Knobs: configs/prompt.yaml in this package (default; M3OPD_PROMPT_CFG overrides).

Native Qwen3.5 chat template, no system prompt. Thinking is on for every domain.
The answer is one \\boxed{} after </think>; one format line per task_type ends the user turn.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CFG = Path(__file__).resolve().parent / "configs" / "prompt.yaml"
TASK_TYPES = ("mc", "math", "count", "short", "ocr", "ground", "tground", "if", "chat", "caption")

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


@functools.cache
def load_cfg(path: str | None = None) -> dict[str, Any]:
    """Load the prompt config. Env M3OPD_PROMPT_CFG overrides the default path."""
    p = Path(path or os.environ.get("M3OPD_PROMPT_CFG") or DEFAULT_CFG)
    cfg = yaml.safe_load(p.read_text())
    missing = [t for t in TASK_TYPES if t not in cfg.get("format_lines", {})]
    if missing:
        raise ValueError(f"{p}: format_lines missing task types {missing}")
    return cfg


FORMAT_LINES: dict[str, str] = dict(load_cfg()["format_lines"])


@functools.cache
def _strip_res(path: str | None = None) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.I | re.M) for p in load_cfg(path).get("strip_patterns") or [])


def format_line(task_type: str, cfg_path: str | None = None) -> str:
    lines = load_cfg(cfg_path)["format_lines"]
    if task_type not in lines:
        raise KeyError(f"unknown task_type {task_type!r}; allowed: {sorted(lines)}")
    return lines[task_type] or ""


def strip_format_instructions(question: str, cfg_path: str | None = None) -> str:
    """Remove answer-format instructions that the source already put in the question."""
    text = question
    for rx in _strip_res(cfg_path):
        text = rx.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def build_user_text(question: str, task_type: str, cfg_path: str | None = None) -> str:
    """Question without source format instructions + the format line for task_type."""
    line = format_line(task_type, cfg_path)
    text = strip_format_instructions(question, cfg_path)
    if not line:
        return text
    return f"{text}{load_cfg(cfg_path).get('separator', chr(10) * 2)}{line}" if text else line


def thinking_for(domain: str | None, task_type: str | None = None, meta: dict | None = None,
                 cfg_path: str | None = None) -> bool:
    """enable_thinking for a row: meta.thinking > domain table > thinking_off_task_types > default.
    thinking_off_task_types absent = ["if"]."""
    if meta and isinstance(meta.get("thinking"), bool):
        return meta["thinking"]
    cfg = load_cfg(cfg_path)
    table = cfg.get("thinking") or {}
    if domain in table:
        return bool(table[domain])
    if task_type in cfg.get("thinking_off_task_types", ["if"]):
        return False
    return bool(cfg.get("thinking_default", True))


def to_boxed(text: str) -> str:
    """Convert '<answer>X</answer>' to '\\boxed{X}'. X that already holds a \\boxed{} is kept as is (no nested
    boxes); converters drop rows whose final answer does not have exactly one box."""
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        if "\\boxed" in inner:
            return inner
        return "\\boxed{" + inner + "}"
    return _ANSWER_TAG.sub(repl, text)


def apply_to_messages(messages: list[dict], task_type: str, cfg_path: str | None = None) -> list[dict]:
    """Return a copy of the messages with build_user_text applied to the last text item of the last
    user turn. Content is a list of {type, text|path} items or a plain string."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = build_user_text(content, task_type, cfg_path)
            return out
        items = [dict(c) for c in content or []]
        for c in reversed(items):
            if c.get("type") == "text":
                c["text"] = build_user_text(c.get("text") or "", task_type, cfg_path)
                break
        else:
            line = format_line(task_type, cfg_path)
            if line:
                items.append({"type": "text", "text": line})
        m["content"] = items
        return out
    raise ValueError("messages contain no user turn")
