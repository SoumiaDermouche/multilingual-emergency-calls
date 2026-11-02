"""Shared helpers for the research questions: corpus, speaker split, text normalisation, statistics."""

import re

import pandas as pd
from scipy.stats import spearmanr

from corpus import ROOT, load_corpus
from pre_annotation import LANGUAGES, normalize

RESULTS_DIR = ROOT / "resultats"
SPEAKERS = ("Operator", "Caller")
ALL = "all"  # pseudo-speaker: the whole call

_SPEAKER = re.compile(r"^(Operator|Caller):\s*")
# the system produced nothing for this turn: counts as an empty hypothesis, not as words
_NOT_TRANSCRIBED = "[non transcrit]"


def split_speakers(text):
    """Text of each speaker (turns joined by newlines) and of the whole call; unlabelled lines continue the turn."""
    parts = {s: [] for s in SPEAKERS}
    speaker = None
    for line in text.split("\n"):
        m = _SPEAKER.match(line)
        if m:
            speaker = m.group(1)
            line = line[m.end():]
        line = line.replace(_NOT_TRANSCRIBED, "").strip()
        if speaker and line:
            parts[speaker].append(line)
    out = {s: "\n".join(lines) for s, lines in parts.items()}
    out[ALL] = "\n".join(
        _SPEAKER.sub("", l).replace(_NOT_TRANSCRIBED, "").strip() for l in text.split("\n")
    ).strip()
    return out


def count_turns(text):
    return sum(bool(_SPEAKER.match(l)) for l in text.split("\n"))


def study_corpus():
    """The calls of the study (four languages, automatic and manual versions present)."""
    df = load_corpus()
    df = df[df.langue.isin(LANGUAGES)]
    return df[(df.transcription_auto != "") & (df.transcription_manuelle != "")].reset_index(drop=True)


def text_language(layer, speaker, call_language):
    """Language code of a text: the operator speaks French, the caller the call language; translation swaps them."""
    code = LANGUAGES[call_language]
    if speaker == ALL:
        return None  # mixed
    spoken = "fr" if speaker == "Operator" else code
    if layer == "transcription":
        return spoken
    return code if spoken == "fr" else "fr"


def spearman_matrix(df, columns):
    """Spearman rho, raw and Holm-corrected p-values for every pair of columns (long format)."""
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            pair = df[[a, b]].dropna()
            rho, p = spearmanr(pair[a], pair[b]) if len(pair) > 2 else (float("nan"), float("nan"))
            rows.append({"metric_a": a, "metric_b": b, "n": len(pair), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    ok = out.p.notna()
    out.loc[ok, "p_holm"] = holm(out.loc[ok, "p"])
    return out


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = pd.Series(p, dtype=float)
    order = p.sort_values().index
    m = len(p)
    adjusted = (p[order] * range(m, 0, -1)).cummax().clip(upper=1)
    return adjusted.reindex(p.index)


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
