"""Per-task_type accuracy scorers for the rule reward. Each returns acc in [0, 1].

Robust matching: one answer cleanup for prediction and gold (clean_answer), units, relation guard, multi-part
answers, option-letter sets, OCR digit guard, more box / point / segment formats, matched multi-box and multi-segment
scores. Config keys that change scores: types.ground.aggregate, types.tground.multi_pred, types.ocr.digit_guard.

Ported parts (Apache-2.0, Vero vero-rl/vero_reward, zlab-princeton): text normalization
(text_normalization.py), balanced \\boxed{} extraction (math_verify_reward_type_boxed.py), relative
numeric tolerance (grade_numeric), IoU matrix + Hungarian component score (grounding_reward.py) and the
instruction-following loop (_instruction_following_reward). Changes: our \\boxed{} answer format replaces
<answer> tags, math_verify off the main thread runs in a process pool (its SIGALRM timeouts need a main
thread), boxes in one 0-1000 grid (no image I/O).
"""
from __future__ import annotations

import ast
import atexit
import concurrent.futures as cf
import json
import multiprocessing as mp
import re
import sys
import threading
import unicodedata
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- text normalization (Vero)
_NON_ALNUM = re.compile(r"[\W_]+")
_WS = re.compile(r"\s+")
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TRAILING_DECIMAL = re.compile(r"\b(\d+)\.0+\b(?!\.)")
_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
                        "\u2014": "-", "\u2026": "...", "\u00a0": " "})


def normalize_text(text: Any) -> str:
    """NFKC, strip accents, unify quotes/dashes, drop thousands separators, lowercase, remove
    punctuation and articles (Vero normalize_text_for_match)."""
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", str(text)))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").translate(_PUNCT)
    s = _TRAILING_DECIMAL.sub(r"\1", _THOUSANDS_SEP.sub("", s)).lower()
    s = _WS.sub(" ", _NON_ALNUM.sub(" ", s)).strip()
    kept = [t for t in s.split() if t not in _ARTICLES]
    if kept and any(len(t) > 1 for t in kept):
        return " ".join(kept)
    return s


# --------------------------------------------------------------------------- \boxed{} extraction (Vero)
def _strip_text_wrapper(text: str) -> str:
    """'\\text{X}' -> 'X' when the wrapper covers the whole payload."""
    s = text.strip()
    m = re.match(r"\\(?:text|textbf|mathrm|mbox)\s*\{", s)
    if not m:
        return text
    i, depth = m.end(), 1
    while i < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
    if depth or s[i:].strip():
        return text
    return s[m.end():i - 1].strip()


def extract_boxed(text: str) -> list[str]:
    """Contents of every balanced \\boxed{...} in order (Vero _extract_boxed_contents)."""
    out: list[str] = []
    idx = 0
    while text:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cur = start + len(r"\boxed")
        while cur < len(text) and text[cur].isspace():
            cur += 1
        if cur >= len(text) or text[cur] != "{":
            idx = cur
            continue
        cur += 1
        depth, begin = 1, cur
        while cur < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[cur], 0)
            cur += 1
        if depth:
            break
        out.append(_strip_text_wrapper(text[begin:cur - 1].strip()))
        idx = cur
    return out


# --------------------------------------------------------------------------- helpers
def _literal(value: Any) -> Any:
    """Parse JSON / Python literal strings; pass other values through (numpy arrays -> lists)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if not isinstance(value, str):
        return value
    s = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
        except Exception:
            continue
        if isinstance(parsed, str) and parsed.strip() != s:
            return _literal(parsed)
        return parsed
    return value


def _aliases(gt: Any) -> list[str]:
    """Ground truth as a list of acceptable strings. Only an explicit list gives several aliases: a Python
    list/tuple, or a str that starts with '[' and parses to a list of strings. Plain text with commas
    ('3, 2, 1', '1,830,000', '(12, 5)') stays one alias."""
    v = gt.tolist() if hasattr(gt, "tolist") else gt
    if isinstance(v, str) and v.strip().startswith("["):
        parsed = _literal(v)
        if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
            v = parsed
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)] if str(v).strip() else []


def _to_float(text: Any) -> float | None:
    try:
        return float(str(text).replace(",", "").strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- answer cleanup
# One cleanup for predictions and golds alike, so a format difference never decides the score: markdown /
# math delimiters, quotes, a leading "The answer is", a trailing period, LaTeX spacing, \text{}-style wrappers
# anywhere in the string, one spelling for percent and degrees.
_WRAP_CMD = re.compile(r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|mathsf|mbox|operatorname|boldsymbol|bm)\s*\{")
_LEAD = re.compile(r"^\s*(?:(?:so|thus|therefore|hence)\s*,?\s+)?(?:the\s+)?(?:final\s+|correct\s+)?"
                   r"(?:answer|result|value|count|number|text|option|choice)\s*(?:is|=|:|：)\s*", re.I)
_LEAD_IT = re.compile(r"^\s*(?:it|this|that|they)\s+(?:is|are|was|reads)\s+", re.I)
_DEG = re.compile(r"(?:\{\s*\})?\^\s*\{?\s*\\circ\s*\}?|\\degree\b|º")
_TEX_SPACE = re.compile(r"\\[,;:!> ]|\\q?quad\b")
_PAIRS = (("**", "**"), ("__", "__"), ("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"), ('"', '"'),
          ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"), ("`", "`"))


def _unwrap_cmds(s: str) -> str:
    """'\\text{ cm}^2' -> ' cm^2': drop the command, keep the braced content (balanced), anywhere in s."""
    while (m := _WRAP_CMD.search(s)):
        i, depth = m.end(), 1
        while i < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[i], 0)
            i += 1
        if depth:
            break
        s = s[:m.start()] + s[m.end():i - 1] + s[i:]
    return s


def clean_answer(text: Any, keep_dollar: bool = False) -> str:
    """Canonical surface form of an answer string (see the section comment). keep_dollar: '$' inside the
    string is kept (OCR currency); a pair that wraps the whole string is still removed."""
    s = str(text).strip()
    if "\\boxed" in s and (b := extract_boxed(s)):
        s = b[-1]
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)   # literal '\\u00b0'
    s = _unwrap_cmds(s.replace("\\left", "").replace("\\right", "").replace("\\!", ""))
    s = s.replace("\\%", "%").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\u2212", "-")
    s = _DEG.sub("°", s)
    s = re.sub(r"(\d)\s*\^\s*\{?\s*(st|nd|rd|th)\s*\}?", r"\1\2", s)   # '12^{th}' -> '12th'
    s = _TEX_SYM.sub(lambda m: _SYM[m.group(1)], s)
    s = _TEX_SPACE.sub(" ", s)
    for _ in range(4):
        prev = s
        s = _LEAD.sub("", _LEAD_IT.sub("", s)).strip()
        s = re.sub(r"^=\s*(?=\S)", "", s)                     # '$=(4,0)$' -> '(4,0)'
        s = re.sub(r"(\d)\.(?=\s*[A-Za-z])", r"\1", s)      # '216. cm^2' -> '216 cm^2'
        for a, b in _PAIRS:
            if len(s) > len(a) + len(b) and s.startswith(a) and s.endswith(b):
                s = s[len(a):len(s) - len(b)].strip()
        s = re.sub(r"[.,;:!。]+$", "", s).strip() or s   # '!' alone stays
        s = re.sub(r"^\*\*|\*\*$", "", s).strip() or s  # unbalanced bold ('**(B)')
        if s == prev:
            break
    if not keep_dollar:
        s = s.replace("$", "").replace("~", " ")   # '~' = LaTeX non-breaking space ('\\mathrm{~cm}')
    return _WS.sub(" ", s).strip()


_SYM = dict(zip("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho sigma "
                "tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega AA".split(),
                "αβγδεεζηθθικλμνξπρστυφφχψωΓΔΘΛΞΠΣΦΨΩÅ"))
_TEX_SYM = re.compile(r"\\(" + "|".join(sorted(_SYM, key=len, reverse=True)) + r")(?![A-Za-z])")
_REL = re.compile(r"\\(?:geq?|leq?|geqslant|leqslant|neq?|gt|lt|gtrsim|lesssim)\b|[<>≤≥≠]")


def _relation_added(pred: str, gold: str) -> bool:
    """The prediction states a relation ('\\geq 5', '> 3') where the gold is a plain value: never a match."""
    return bool(_REL.search(pred)) and not _REL.search(gold)


_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}


def word_number(text: str) -> int | None:
    """'three' -> 3, 'Twenty-one' -> 21, 'one hundred' -> 100 (0..999); anything else None."""
    toks = re.split(r"[\s\-]+", str(text).strip().lower().replace(" and ", " "))
    if not toks or not all(t in _NUM_WORDS or t in _TENS or t == "hundred" for t in toks):
        return None
    total = cur = 0
    for t in toks:
        if t == "hundred":
            cur = max(cur, 1) * 100
        else:
            cur += _NUM_WORDS.get(t, _TENS.get(t, 0))
    return total + cur


# --------------------------------------------------------------------------- mc
_MC_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is)?\s*[:：]?\s*", re.I)
# letter alone or 'B.', 'B)', 'B:', '(B)', 'B - text', 'B, text', 'B = text'
_MC_LETTER = re.compile(r"^\(?\s*([A-Za-z])\s*(?:[.:)]|\s*[-\u2013\u2014=,](?:\s|$)|$)")
_MC_MULTI = re.compile(r"^\(?[A-Z]\)?(?:\s*(?:,|;|/|&|\band\b|\bor\b)?\s*\(?[A-Z]\)?)+$")
_MC_MORE = re.compile(r"(?:,|&|\band\b|\bor\b)\s*\(?([A-H])\)?\s*$")  # 'D. backpack and A', '(D) and (A)'


def mc_letter(text: str) -> str | None:
    """Option letter of a payload. 'I think B' and 'A cat' give None (a word, not an option)."""
    s = _MC_PREFIX.sub("", clean_answer(text))
    m = _MC_LETTER.match(s)
    return m.group(1).upper() if m else None


def mc_letters(text: str, concat: bool = False) -> frozenset[str] | None:
    """Set of option letters: 'B', '(B) and (D)', 'B, D', and 'BD' when concat. A single letter followed by
    option text that ends in another letter ('D. backpack and A') counts both letters. None: not an option."""
    s = _MC_PREFIX.sub("", clean_answer(text)).strip()
    if s and _MC_MULTI.match(s) and len(re.findall(r"[A-Z]", s)) > 1:
        return frozenset(re.findall(r"[A-Z]", s))
    if concat and re.fullmatch(r"[A-Z]{2,8}", s) and len(set(s)) == len(s):
        return frozenset(s)
    one = mc_letter(s)
    if one is None:
        return None
    more = _MC_MORE.search(s[1:].lstrip(").").strip()) if len(s) > 2 else None
    return frozenset({one, more.group(1)}) if more else frozenset({one})


def score_mc(pred: str, gt: Any, cfg: dict) -> float:
    """Exact option-letter set. A gold that is not an option letter (rare data rows: 'AAS', '3') is scored as
    a short answer."""
    for g in _aliases(gt):
        gs = mc_letters(g, concat=True)
        if gs is None:
            if score_short(pred, g, cfg):
                return 1.0
            continue
        if mc_letters(pred, concat=len(gs) > 1) == gs:
            return 1.0
    return 0.0


# --------------------------------------------------------------------------- math
def _mv_parse(text: str, timeout: int | None):
    from math_verify import parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    cfgs = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    return parse("\\boxed{" + text + "}", cfgs, parsing_timeout=timeout)


def _mv_equal(pred: str, gt: str, rounding: int, precision: int, timeout: int | None) -> bool:
    """math_verify parse + verify. Its timeouts use SIGALRM, so this must run in a main thread."""
    from math_verify import verify

    g, p = _mv_parse(gt, timeout), _mv_parse(pred, timeout)
    if not g or not p:
        return False
    return bool(verify(g, p, float_rounding=rounding, numeric_precision=precision, timeout_seconds=timeout))


_POOL: cf.ProcessPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool(workers: int) -> cf.ProcessPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        return _POOL


@atexit.register
def _close_pool() -> None:
    """Shut the pool down before interpreter teardown (avoids a noisy executor weakref callback)."""
    global _POOL
    pool, _POOL = _POOL, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


def _reset_pool(pool: cf.ProcessPoolExecutor) -> None:
    """Kill a pool whose worker hangs (sympy ignored the alarm) so the next call gets fresh workers."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is pool:
            _POOL = None
    for proc in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            proc.kill()
        except Exception:
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def _mv_equal_bounded(pred: str, gt: str, cfg: dict) -> bool:
    """_mv_equal with a wall-clock bound in every thread. verl's reward loop calls compute_score in an
    executor thread, where SIGALRM is not available; there the work goes to a spawn process pool (workers
    run tasks in their main thread, so math_verify's own timeouts apply) and a hard timeout kills it."""
    lim = cfg.get("limits", {})
    mc = cfg.get("types", {}).get("math", {})
    t = int(lim.get("math_timeout_s", 3))
    args = (pred, gt, int(mc.get("float_rounding", 6)), int(mc.get("numeric_precision", 15)), t)
    if threading.current_thread() is threading.main_thread():
        try:
            return _mv_equal(*args)
        except Exception:
            return False
    pool = _pool(int(lim.get("math_workers", 4)))
    try:
        return bool(pool.submit(_mv_equal, *args).result(timeout=float(lim.get("math_hard_timeout_s", 20))))
    except cf.TimeoutError:
        print(f"[m3opd.reward] math_verify hard timeout; pred={pred[:60]!r} scored 0", file=sys.stderr)
        _reset_pool(pool)
    except cf.BrokenExecutor:
        _reset_pool(pool)
    except Exception:
        pass
    return False


def _compact(text: Any) -> str:
    """Lowercase, no whitespace; keeps signs, operators and brackets (unlike normalize_text)."""
    return _WS.sub("", str(text)).lower()


def math_equal(pred: str, gt: str, cfg: dict, tolerance: float | None = None) -> bool:
    """math_verify equivalence of two bare answers, with an optional relative tolerance (Vero)."""
    if _compact(gt) and _compact(pred) == _compact(gt):
        return True
    if tolerance is not None and tolerance >= 0:
        a, b = _to_float(gt), _to_float(pred)
        if a is not None and b is not None:
            return abs(a - b) <= (tolerance if abs(a) <= 1e-12 else tolerance * abs(a))
    return _mv_equal_bounded(pred, gt, cfg)


_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*[eE][-+]?\d+)?|-?\.\d+"
# a number followed by a unit: 'cm', 'cm^2', 'm/s', '°', '%', 'kg', 'square units', 'th' (ordinal)
_NUM_UNIT = re.compile(rf"^\s*({_NUMBER}|\\frac\{{[^{{}}]+\}}\{{[^{{}}]+\}})\s*([A-Za-z°%µμΩ][A-Za-z°%µμΩ\s^/.\-\d{{}}]*?)\s*$")
_UNIT_ALIAS = {"°": "deg", "deg": "deg", "degree": "deg", "degrees": "deg", "rad": "rad", "radian": "rad",
               "radians": "rad", "%": "%", "percent": "%", "unit": "", "units": "", "sq units": "",
               "square units": "", "st": "ord", "nd": "ord", "rd": "ord", "th": "ord"}


def split_unit(s: str) -> tuple[str, str] | None:
    """'24 cm^2' -> ('24', 'cm2'); '45°' -> ('45', 'deg'); '50%' -> ('50', '%'); None if not number+unit."""
    m = _NUM_UNIT.match(s)
    if not m:
        return None
    unit = re.sub(r"[\s{}^]", "", m.group(2).lower()).rstrip(".")
    unit = _UNIT_ALIAS.get(m.group(2).strip().lower(), _UNIT_ALIAS.get(unit, unit))
    if re.fullmatch(r"[a-z]", unit):  # '2x', '3 n': a variable, not a unit
        return None
    return m.group(1).replace(",", "") if re.fullmatch(_NUMBER, m.group(1)) else m.group(1), unit


_LABEL = re.compile(r"^\s*[A-Za-zα-ωΑ-Ω][A-Za-z\s']{0,40}?(?:\([A-Za-z, ]*\))?\s*=\s*(?=\S)")   # 'Area = 5', 'g(x) = 5'


def _strip_label(s: str) -> str:
    return _LABEL.sub("", s) if s.count("=") == 1 else s


_UNITS = ("mm cm dm km m mg kg g ms s min h hr hours minutes seconds ml l L mL cc in ft yd mi inch inches feet "
          "N kN J kJ W kW V mV A mA Pa kPa Hz kHz Ω ohm ohms mol K °C °F units unit sq").split()
_TRAIL_UNIT = re.compile(r"(?<=[\d}\)πa-zA-Z°])\s*(?:" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True))
                         + r")(?:\s*\^\s*\{?\s*[23]\s*\}?|²|³)?\s*$")


def _strip_unit(s: str) -> str:
    """'121 \\pi cm^2' -> '121 \\pi' (a unit after an expression); keeps a bare unit word or variable alone."""
    t = _TRAIL_UNIT.sub("", s).strip()
    return t if t and t != s and re.search(r"[\d\\]", t) else s


def _expand_pm(s: str) -> list[str] | None:
    """'1 \\pm \\sqrt{19}' -> ['1+\\sqrt{19}', '1-\\sqrt{19}'] (one top-level \\pm / ±), else None."""
    parts = re.split(r"\\pm|±", s)
    return [parts[0] + "+" + parts[1], parts[0] + "-" + parts[1]] if len(parts) == 2 and parts[0].strip() else None


_SEP = re.compile(r"\s*[,;\n]\s*|\s+(?:and|or)\s+")


def split_parts(s: str) -> list[str]:
    """Top-level parts of a multi-part answer: separators ',', ';', newline, ' and ', ' or ', '\\to', '→',
    outside brackets. '1,830,000' and '(1, 2)' stay one part."""
    s = s.strip()
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        return [s]
    s = re.sub(r"\s*(?:\\to|→|->|\\rightarrow)\s*", ",", s)
    parts, depth, start, i = [], 0, 0, 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and (m := _SEP.match(s, i)):
            parts.append(s[start:i])
            i = start = m.end()
            continue
        i += 1
    parts.append(s[start:])
    return [x.strip() for x in parts if x.strip()] or [s]


def _skeleton(s: str) -> tuple:
    """Shape (the compact string with every number replaced by '#') and the numbers as floats, in order: an
    answer that math_verify calls equal to a gold it can only parse in part must also match this."""
    c = _compact(s)
    nums = re.findall(r"\d+(?:\.\d+)?", c)
    return re.sub(r"\d+(?:\.\d+)?", "#", c), tuple(float(n) for n in nums)


def _texty(gold: str) -> bool:
    """Words or non-ASCII symbols outside LaTeX commands ('inches', 'α', '∠'): math_verify parses such a gold
    only in part."""
    return bool(re.search(r"[^\x00-\x7f°]|[A-Za-z]{3,}", re.sub(r"\\[A-Za-z]+", "", gold)))


def _risky(gold: str) -> bool:
    return _texty(gold)


def _atom_match(pred: str, gold: str, cfg: dict, tol: float | None) -> bool:
    """One-part answers: text equality, number words, number + unit (units must agree when both are given;
    a missing unit is accepted), then math_verify (guarded by _skeleton for golds it cannot fully parse)."""
    p, g = _strip_label(pred), _strip_label(gold)
    if _compact(p) and _compact(p) == _compact(g):
        return True
    if not (split_unit(p) or split_unit(g)):
        p, g = _strip_unit(p), _strip_unit(g)
    if (q := _APPROX.sub("", p)) != p and re.match(r"[-+]?\.?\d", q):   # 'about 3' -> '3'
        p = q
    p = str(w) if (w := word_number(p)) is not None else p
    g = str(w) if (w := word_number(g)) is not None else g
    pu, gu = split_unit(p), split_unit(g)
    if pu or gu:
        (pn, punit), (gn, gunit) = pu or (p, ""), gu or (g, "")
        if pu and not gu and (m := re.fullmatch(rf"({_NUMBER})\s*([a-z])", g)):   # '12s' vs '12 ms'
            gn, gunit = m.group(1), m.group(2)
        if punit != gunit and punit and gunit and "ord" not in (punit, gunit):
            return False
        if math_equal(pn, gn, cfg, tol):
            return True
        pf, gf = _to_float(pn), _to_float(gn)
        if pf is not None and gf is not None and "%" in (punit, gunit) and not (punit and gunit):
            pf, gf = (pf * 100, gf) if gunit == "%" else (pf, gf * 100)   # '0.4' vs '40%'
            return abs(pf - gf) <= 1e-6 * max(1.0, abs(gf))
        return False
    if not math_equal(p, g, cfg, tol):
        if "°" in g and "°" not in p and math_equal(p, g.replace("°", ""), cfg, tol):   # '(315, 1)' vs '(315°, 1)'
            return True
        return False
    return not _risky(g) or _compact(p) == _compact(g) or _skeleton(p) == _skeleton(g)


def math_match(pred: str, gold: str, cfg: dict, tol: float | None = None) -> bool:
    """Math answer match (see the cleanup section): relation guard, option-letter golds, multi-part
    answers part by part (same count; any order only when every gold part is an equation such as 'x=1')."""
    p, g = clean_answer(pred), clean_answer(gold)
    if not p or not g:
        return False
    if _compact(p) == _compact(g):
        return True
    if _relation_added(p, g):
        return False
    if re.fullmatch(r"\(?[A-Z]\)?", g):
        return mc_letters(p) == frozenset(g.strip("()"))
    gp, pp = split_parts(g), split_parts(p)
    if len(gp) == 1 and len(pp) == 2 and (pm := _expand_pm(g)):   # '1 \\pm \\sqrt{19}' vs '1-\\sqrt{19}, 1+\\sqrt{19}'
        gp = pm
    if len(gp) > 1 or len(pp) > 1:
        if len(gp) != len(pp):
            return False
        if len(gp) == 2 and _expand_pm(g):   # \\pm parts in any order
            return any(all(_atom_match(a, b, cfg, tol) for a, b in zip(o, gp)) for o in (pp, pp[::-1]))
        if all(_atom_match(a, b, cfg, tol) for a, b in zip(pp, gp)):
            return True
        if all("=" in x for x in gp):
            left = list(pp)
            for b in gp:
                hit = next((a for a in left if _atom_match(a, b, cfg, tol)), None)
                if hit is None:
                    return False
                left.remove(hit)
            return True
        return False
    return _atom_match(p, g, cfg, tol)


def score_math(pred: str, gt: Any, cfg: dict, extra_info: dict | None = None) -> float:
    tol = (extra_info or {}).get("tolerance")
    tol = _to_float(tol) if tol is not None else None
    return 1.0 if any(math_match(pred, g, cfg, tol) for g in _aliases(gt)) else 0.0


# --------------------------------------------------------------------------- count
_APPROX = re.compile(r"^(?:about|approximately|approx\.?|around|roughly|~|≈|\\approx)\s*", re.I)


def parse_int(text: str) -> int | None:
    """'7', 'seven', '7 birds', 'about 7', '**7**' -> 7. A relation ('> 7'), a range or a second number: None."""
    s = _APPROX.sub("", clean_answer(text))
    if _REL.search(s):
        return None
    if (w := word_number(s)) is not None:
        return w
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    m = re.fullmatch(r"([-+]?\d+)(?:\.0+)?(?:\s+[A-Za-z][A-Za-z\s'-]*)?", s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"([A-Za-z]+(?:-[A-Za-z]+)?)\s+[A-Za-z][A-Za-z\s'-]*", s)
    if m and (w := word_number(m.group(1))) is not None:
        return w
    return None


def score_count(pred: str, gt: Any, cfg: dict) -> float:
    """Integer match. A gold that is not one integer (rare rows: '5, 5') is scored as a short answer."""
    p = parse_int(pred)
    for g in _aliases(gt):
        gi = parse_int(g)
        if gi is None:
            if score_short(pred, g, cfg):
                return 1.0
        elif p is not None and p == gi:
            return 1.0
    return 0.0


# --------------------------------------------------------------------------- short
def short_match(pred: str, gold: str, cfg: dict) -> bool:
    p, g = clean_answer(pred), clean_answer(gold)
    if not p or _relation_added(p, g):
        return False
    if re.fullmatch(r"\(?[A-Za-z]\)?", g):  # a letter gold ('W', '(c)'): option-letter forms count
        return mc_letters(p) == frozenset(g.strip("()").upper())
    numeric = cfg.get("types", {}).get("short", {}).get("numeric_fallback", True)
    if _to_float(g) is not None or split_unit(g) or word_number(g) is not None:
        return numeric and math_match(p, g, cfg)   # a number: sign, unit and number words matter
    gp, pp = split_parts(g), split_parts(p)
    if len(gp) > 1 and all(re.fullmatch(r"\(?[A-Z]\)?", x) for x in gp):   # a set of option letters: any order
        return sorted(x.strip("()") for x in gp) == sorted(x.strip("()").upper() for x in pp)
    if len(gp) > 1 and len(gp) == len(pp) and all(short_match(a, b, cfg) for a, b in zip(pp, gp)):
        return True
    np_, ng = normalize_text(p), normalize_text(g)
    if np_ and (np_ == ng or np_.replace(" ", "") == ng.replace(" ", "")):   # 'T_4' vs 'T4'
        return True
    if not ng and _compact(g) and _compact(p) == _compact(g):
        return True
    return bool(numeric and re.search(r"\d", g) and not _texty(g) and math_match(p, g, cfg))


def score_short(pred: str, gt: Any, cfg: dict) -> float:
    """Normalized text match (Vero), then the math match for golds with numbers (signs, units, number words)."""
    return 1.0 if any(short_match(pred, g, cfg) for g in _aliases(gt)) else 0.0


# --------------------------------------------------------------------------- ocr (ANLS)
def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gt: str, threshold: float = 0.5) -> float:
    p, g = (" ".join(str(x).lower().split()) for x in (pred, gt))
    if not p and not g:
        return 1.0
    sim = 1.0 - _levenshtein(p, g) / max(len(p), len(g))
    return sim if sim >= threshold else 0.0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def score_ocr(pred: str, gt: Any, cfg: dict) -> float:
    """ANLS on the cleaned strings (quotes, markdown, a trailing period and a leading 'The text is' do not
    count). Digit guard: when the digit sequences differ ('$450.00' vs '$540.00') the score is 0."""
    o = cfg.get("types", {}).get("ocr", {})
    thr, guard = o.get("anls_threshold", 0.5), o.get("digit_guard", False)
    p = _THOUSANDS_SEP.sub("", clean_answer(pred, keep_dollar=True))
    best = 0.0
    for g in _aliases(gt):
        gc = _THOUSANDS_SEP.sub("", clean_answer(g, keep_dollar=True))
        if guard and _digits(gc) != _digits(p):
            continue
        best = max(best, anls(p, gc, thr))
    return best


# --------------------------------------------------------------------------- ground (Vero grounding_reward)
_NUM = r"-?\d+(?:\.\d+)?"
_BOX4 = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")


_PAIR2 = re.compile(rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*,?\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)")
_KV4 = re.compile(rf"x_?1\s*[=:]\s*({_NUM}).*?y_?1\s*[=:]\s*({_NUM}).*?x_?2\s*[=:]\s*({_NUM}).*?y_?2\s*[=:]\s*({_NUM})",
                  re.I | re.S)
_BARE4 = re.compile(rf"^\s*[\[(]?\s*({_NUM})\s*[,\s]\s*({_NUM})\s*[,\s]\s*({_NUM})\s*[,\s]\s*({_NUM})\s*[\])]?\s*$")
_BOX_TAGS = re.compile(r"</?box>|<\|box_(?:start|end)\|>|<\|object_ref_(?:start|end)\|>|\\left|\\right")


def parse_boxes(value: Any) -> list[list[float]]:
    """Boxes from '[x1,y1,x2,y2]', '[[..],[..]]', [{'bbox_2d': [..]}, ..], '(x1,y1),(x2,y2)' (also inside
    <box>/<|box_start|> tags), 'x1=.., y1=.., x2=.., y2=..', four bare numbers, or free text with [a,b,c,d]."""
    v = _literal(value)
    if isinstance(v, dict):
        v = [v]
    if isinstance(v, (list, tuple)):
        if len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
            return [[float(x) for x in v]]
        out = []
        for item in v:
            if isinstance(item, dict):
                item = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            if isinstance(item, (list, tuple)) and len(item) == 4:
                try:
                    out.append([float(x) for x in item])
                except (TypeError, ValueError):
                    continue
        if out:
            return out
    text = _BOX_TAGS.sub(" ", str(value))
    for pat in (_BOX4, _PAIR2, _KV4):
        found = [[float(x) for x in m.groups()] for m in pat.finditer(text)]
        if found:
            return found
    m = _BARE4.match(text)
    return [[float(x) for x in m.groups()]] if m else []


_PT2 = re.compile(rf"[\[(]\s*({_NUM})\s*,\s*({_NUM})\s*[\])]")


def parse_points(value: Any) -> list[list[float]]:
    """Points from '[x, y]', '(x, y)', [[x, y], ..], {'point_2d': [x, y]} or a list of such dicts."""
    v = _literal(value)
    if isinstance(v, dict):
        v = [v]
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            return [[float(v[0]), float(v[1])]]
        out = []
        for item in v:
            if isinstance(item, dict):
                item = item.get("point_2d") or item.get("point")
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                out.append([float(item[0]), float(item[1])])
        if out:
            return out
    return [[float(a), float(b)] for a, b in _PT2.findall(_BOX_TAGS.sub(" ", str(value)))]


def _match_mean(sim: np.ndarray) -> float:
    """Hungarian matching on a similarity matrix: sum of matched similarities / max(n_pred, n_gt), so a
    missing or a spurious item costs its share ('matched' aggregate)."""
    from scipy.optimize import linear_sum_assignment

    if not sim.size:
        return 0.0
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].sum() / max(sim.shape))


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred, gt = np.asarray(pred, float).reshape(-1, 4), np.asarray(gt, float).reshape(-1, 4)
    if not len(pred) or not len(gt):
        return np.zeros((len(pred), len(gt)))
    pa = (pred[:, 2] - pred[:, 0]).clip(min=0) * (pred[:, 3] - pred[:, 1]).clip(min=0)
    ga = (gt[:, 2] - gt[:, 0]).clip(min=0) * (gt[:, 3] - gt[:, 1]).clip(min=0)
    lo, hi = np.maximum(pred[:, None, :2], gt[None, :, :2]), np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    iw, ih = (hi - lo).clip(min=0)[..., 0], (hi - lo).clip(min=0)[..., 1]
    inter = iw * ih
    return inter / np.maximum(pa[:, None] + ga[None, :] - inter, 1e-12)


def grounding_component_score(pred: list, gt: list, weights: dict, iou_thr: float,
                              normalize: bool = True) -> float:
    """Vero: location*mean IoU(TP) + recall*F1 - penalty*0.5*(fp_rate+fn_rate), Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    if not gt:
        return 0.0
    if not pred:
        return 0.0
    ious = iou_matrix(np.array(pred), np.array(gt))
    cost = 1.0 - ious
    cost[ious < iou_thr] = 1e6
    rows, cols = linear_sum_assignment(cost)
    matched = [ious[r, c] for r, c in zip(rows, cols) if ious[r, c] >= iou_thr]
    tp, n, m = len(matched), len(pred), len(gt)
    prec, rec = tp / n, tp / m
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mean_iou = float(np.mean(matched)) if matched else 0.0
    miss = 0.5 * ((n - tp) / n + (m - tp) / m)
    s = weights["location"] * mean_iou + weights["recall"] * f1 - weights["penalty"] * miss
    if normalize:  # Vero clips the raw sum, so 1 of 2 boxes already scores 1.0
        s /= weights["location"] + weights["recall"]
    return float(np.clip(s, 0.0, 1.0))


def _rescale(pred: list[list[float]], gt: list[list[float]], scale: float) -> list[list[float]]:
    """A 0-1 normalised prediction against 0-scale golds: multiply by scale."""
    if pred and gt and max(max(b) for b in pred) <= 1.0 < max(max(b) for b in gt):
        return [[x * scale for x in b] for b in pred]
    return pred


def score_point(pred: str, gt_points: list[list[float]], cfg: dict) -> float:
    """Point golds: a predicted point within point_radius (0-scale units) of a gold point, or a predicted box
    that contains it, scores 1; matched across several points like boxes."""
    g = cfg.get("types", {}).get("ground", {})
    scale, radius = float(g.get("coord_scale", 1000)), float(g.get("point_radius", 20))
    pts = _rescale(parse_points(pred), gt_points, scale)
    boxes = [] if pts else _rescale(parse_boxes(pred), gt_points, scale)
    if pts:
        sim = np.array([[float(np.hypot(p[0] - q[0], p[1] - q[1]) <= radius) for q in gt_points] for p in pts])
    elif boxes:
        sim = np.array([[float(b[0] <= q[0] <= b[2] and b[1] <= q[1] <= b[3]) for q in gt_points] for b in boxes])
    else:
        return 0.0
    return _match_mean(sim)


def score_ground(pred: str, gt: Any, cfg: dict, extra_info: dict | None = None) -> float:
    g = cfg.get("types", {}).get("ground", {})
    scale = float(g.get("coord_scale", 1000))
    lit = _literal(gt)
    if isinstance(lit, (list, tuple)) and lit and (all(isinstance(x, (list, tuple)) and len(x) == 2 for x in lit)
                                                   or (len(lit) == 2 and all(isinstance(x, (int, float)) for x in lit))):
        return score_point(pred, parse_points(gt), cfg)   # point golds '[(x, y), ..]', '[x, y]'
    gt_boxes = parse_boxes(gt)
    size = (extra_info or {}).get("image_size")
    if (extra_info or {}).get("gt_is_pixel") and size:  # convert pixel gt to the 0-scale grid
        w, h = _literal(size)[:2]
        gt_boxes = [[b[0] / w * scale, b[1] / h * scale, b[2] / w * scale, b[3] / h * scale]
                    for b in gt_boxes]
    if not gt_boxes or all(b[2] <= b[0] or b[3] <= b[1] for b in gt_boxes):   # point golds / zero-area boxes
        pts = parse_points(gt) or [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in gt_boxes]
        return score_point(pred, pts, cfg) if pts else 0.0
    pred_boxes = _rescale(parse_boxes(pred), gt_boxes, scale)
    if not pred_boxes:
        return 0.0
    if len(gt_boxes) == 1 and len(pred_boxes) == 1:
        return float(iou_matrix(np.array(pred_boxes), np.array(gt_boxes))[0, 0])
    if g.get("aggregate", "vero") == "matched":
        return _match_mean(iou_matrix(np.array(pred_boxes), np.array(gt_boxes)))
    return grounding_component_score(pred_boxes, gt_boxes, g.get("weights") or
                                     {"location": 1.0, "recall": 0.75, "penalty": 0.5},
                                     float(g.get("iou_threshold", 0.5)), bool(g.get("normalize", True)))


# --------------------------------------------------------------------------- tground
_TS = r"(?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?"
_SEG = re.compile(rf"({_TS})\s*(?:s|sec|secs|seconds?)?\s*(?:,|-{{1,2}}|–|—|~|→|->|\\to|to|and|until)\s*({_TS})", re.I)
_SEG_KV = re.compile(rf"start\w*[\"']?\s*[:=]?\s*({_TS}).*?end\w*[\"']?\s*[:=]?\s*({_TS})", re.I | re.S)


def _secs(t: Any) -> float:
    """'83.5' -> 83.5, '1:23.5' -> 83.5, '1:02:03' -> 3723."""
    out = 0.0
    for part in str(t).split(":"):
        out = out * 60 + float(part)
    return out


def parse_segments(value: Any) -> list[tuple[float, float]]:
    """Segments from '[s, e]', [[s, e], ..], {'start': s, 'end': e}, 'start: s, end: e', 's-e', 's to e',
    'between s and e', 'mm:ss - mm:ss' and the separators ~ -- → \\to."""
    v = _literal(value)
    if isinstance(v, dict):
        v = [v]
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            return [(float(v[0]), float(v[1]))]
        out = []
        for s in v:
            if isinstance(s, dict) and "start" in s and "end" in s:
                s = (s["start"], s["end"])
            if isinstance(s, (list, tuple)) and len(s) == 2 and all(isinstance(x, (int, float)) for x in s):
                out.append((float(s[0]), float(s[1])))
        if out:
            return out
    text = re.sub(r"\\[,;: ]", " ", _unwrap_cmds(str(value)))
    found = [(_secs(a), _secs(b)) for a, b in _SEG.findall(text)] or \
            [(_secs(a), _secs(b)) for a, b in _SEG_KV.findall(text)]
    return found


def tiou(p: tuple[float, float], g: tuple[float, float]) -> float:
    if p[1] < p[0] or g[1] < g[0]:
        return 0.0
    inter = max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
    union = max(p[1], g[1]) - min(p[0], g[0])
    return inter / union if union > 0 else float(p == g)


def score_tground(pred: str, gt: Any, cfg: dict) -> float:
    """One predicted segment: best tIoU over the gold segments (several golds = any is right). Several
    predicted segments: matched tIoU / max(n_pred, n_gold), so extra segments cost their share."""
    ps, gs = parse_segments(pred), parse_segments(gt)
    if not ps or not gs:
        return 0.0
    if len(ps) == 1:
        return max(tiou(ps[0], g) for g in gs)
    if cfg.get("types", {}).get("tground", {}).get("multi_pred", False):
        return _match_mean(np.array([[tiou(p, g) for g in gs] for p in ps]))
    return 0.0


# --------------------------------------------------------------------------- if (not supported)
def parse_constraints(gt: Any) -> list[tuple[str, dict]]:
    """Accepts {'instruction_id_list': [...], 'kwargs': [...]}, {'instructions': [{'class'|'id', 'args'|
    'kwargs'}]} (Vero) or a list of such items. Returns [(id_or_class, kwargs)]."""
    v = _literal(gt)
    if isinstance(v, dict) and "instruction_id_list" in v:
        ids = _literal(v["instruction_id_list"])
        kws = _literal(v.get("kwargs"))
        if kws is None:
            kws = [{}] * len(ids)
        if not isinstance(kws, (list, tuple)) or len(kws) != len(ids):
            raise ValueError(f"if gt: {len(ids)} instruction ids but kwargs {kws!r:.80}")
        return [(str(i), dict(_literal(k) or {})) for i, k in zip(ids, kws)]
    if isinstance(v, dict):
        v = v.get("instructions") or v.get("constraints") or []
    out = []
    for item in v if isinstance(v, (list, tuple)) else []:
        if isinstance(item, str):
            out.append((item, {}))
        elif isinstance(item, dict):
            name = item.get("instruction_id") or item.get("id") or item.get("class")
            if name:
                out.append((str(name), dict(_literal(item.get("kwargs") or item.get("args") or {}) or {})))
    return out


def score_if(response: str, gt: Any, cfg: dict) -> float:
    """Instruction-following scorer: not part of this release (it needs the IFEval checker library)."""
    raise NotImplementedError("task_type 'if' (instruction following) is not supported in this release")


BOXED_SCORERS = {"mc": score_mc, "math": score_math, "count": score_count, "short": score_short,
                 "ocr": score_ocr, "ground": score_ground, "tground": score_tground}
NEEDS_EXTRA = {"math", "ground"}
