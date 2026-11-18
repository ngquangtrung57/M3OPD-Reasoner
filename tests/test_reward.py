"""Tests for m3opd.reward. Pure CPU, no model."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from m3opd.reward import types as T  # noqa: E402
from m3opd.reward.score import compute_score  # noqa: E402


def think(answer, reasoning="Let me look at the image carefully."):
    return f"{reasoning}\n</think>\n\n{answer}"


def score(task_type, response, gt, **info):
    return compute_score("test", response, gt, {"task_type": task_type, **info})


def assert_r(r, acc, fmt):
    assert r["acc"] == pytest.approx(acc, abs=1e-6)
    assert r["fmt"] == pytest.approx(fmt)
    assert r["score"] == pytest.approx(0.9 * r["acc"] + 0.1 * r["fmt"])


# (task_type, correct response, wrong response, ground truth, extra_info)
CASES = [
    ("mc", think("The answer is \\boxed{B}."), think("\\boxed{C}"), "B", {}),
    ("math", think("So x = \\boxed{\\frac{3}{4}}"), think("\\boxed{0.7}"), "0.75", {}),
    ("count", think("There are \\boxed{7} birds."), think("\\boxed{6}"), "7", {}),
    ("short", think("\\boxed{The Eiffel Tower}"), think("\\boxed{Big Ben}"), "eiffel tower", {}),
    ("ocr", think("\\boxed{Total: 42.50 USD}"), think("\\boxed{hello world}"), "TOTAL: 42.50 USD", {}),
    ("ground", think("\\boxed{[100, 200, 300, 400]}"), think("\\boxed{[600, 600, 700, 700]}"),
     "[100, 200, 300, 400]", {}),
    ("tground", think("\\boxed{[12.0, 20.0]}"), think("\\boxed{[40, 50]}"), "[12, 20]", {}),
]


@pytest.mark.parametrize("task_type,good,bad,gt,info", CASES, ids=[c[0] for c in CASES])
def test_correct(task_type, good, bad, gt, info):
    assert_r(score(task_type, good, gt, **info), 1.0, 1.0)


@pytest.mark.parametrize("task_type,good,bad,gt,info", CASES, ids=[c[0] for c in CASES])
def test_wrong(task_type, good, bad, gt, info):
    assert_r(score(task_type, bad, gt, **info), 0.0, 1.0)


@pytest.mark.parametrize("task_type,good,bad,gt,info", CASES, ids=[c[0] for c in CASES])
def test_format_fail_no_think_close(task_type, good, bad, gt, info):
    truncated = good.replace("</think>", "")          # truncated / never closed the think block
    assert_r(score(task_type, truncated, gt, **info), 0.0, 0.0)


@pytest.mark.parametrize("task_type,good,bad,gt,info", CASES, ids=[c[0] for c in CASES])
def test_format_two_boxes_zeroes_acc(task_type, good, bad, gt, info):
    hedge = good + " or maybe \\boxed{0}"
    r = score(task_type, hedge, gt, **info)
    assert_r(r, 0.0, 0.5)


@pytest.mark.parametrize("task_type,good,bad,gt,info", CASES, ids=[c[0] for c in CASES])
def test_format_no_box(task_type, good, bad, gt, info):
    plain = think(T.extract_boxed(good)[0])
    assert_r(score(task_type, plain, gt, **info), 0.0, 0.5)


# ---------------------------------------------------------------- format details
def test_leading_think_tag_and_special_tokens_ok():
    r = score("mc", "<think>\nreason\n</think>\n\\boxed{A}<|im_end|>", "A")
    assert_r(r, 1.0, 1.0)


def test_empty_reasoning_fails():
    assert_r(score("mc", "</think>\\boxed{A}", "A"), 0.0, 0.0)


def test_box_inside_reasoning_only_fails():
    assert_r(score("mc", "maybe \\boxed{A}\n</think>\nA", "A"), 0.0, 0.5)


def test_two_think_blocks_fail():
    assert_r(score("mc", "a</think>b</think>\\boxed{A}", "A"), 0.0, 0.0)


def test_nothink_mode_by_flag():
    assert_r(score("mc", "\\boxed{A}", "A", thinking=False), 1.0, 1.0)
    assert_r(score("mc", "x</think>\\boxed{A}", "A", thinking=False), 0.0, 0.0)


def test_unknown_task_type_scores_zero():
    r = score("essay", think("\\boxed{A}"), "A")
    assert r["score"] == 0.0 and r["task_type"] == "essay"


def test_return_keys():
    assert set(score("mc", think("\\boxed{A}"), "A")) == {"score", "acc", "fmt", "task_type"}


# ---------------------------------------------------------------- per-type details
@pytest.mark.parametrize("pred,gt,want", [("(B)", "B", 1), ("Option B", "B", 1), ("b.", "(B)", 1),
                                          ("\\text{B}", "B", 1), ("A", "B", 0), ("cat", "B", 0)])
def test_mc_letters(pred, gt, want):
    assert T.score_mc(pred, gt, {}) == want


@pytest.mark.parametrize("pred,gt,want", [("2\\sqrt{2}", "\\sqrt{8}", 1), ("x=2", "2", 1), ("1/2", "0.5", 1),
                                          ("3", "4", 0), ("", "4", 0)])
def test_math_equivalence(pred, gt, want):
    assert T.score_math(pred, gt, {}) == want


def test_math_tolerance():
    assert T.score_math("101", "100", {}, {"tolerance": 0.02}) == 1
    assert T.score_math("110", "100", {}, {"tolerance": 0.02}) == 0



def test_if_task_type_is_not_supported():
    with pytest.raises(NotImplementedError):
        T.score_if("x", {}, {})
    assert score("if", think("anything"), {})["acc"] == 0.0
