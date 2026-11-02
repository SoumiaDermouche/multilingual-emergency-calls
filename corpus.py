"""Corpus loading: data/corpus.csv, one row per call.

Columns:
  - call, langue, scenario, categorie_urgence
  - transcription_auto, transcription_manuelle    automatic (Whisper) and manual (post-edited) transcription
  - traduction_auto, traduction_manuelle          automatic (LLM) and manual translation
  - appelant_*, operateur_*                       questionnaire scores (1-5) of the caller and of the operator

Speaker labels are normalised on save: every line starts with "Operator:" or "Caller:".
"""

import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
CORPUS_FILE = DATA_DIR / "corpus.csv"

# questionnaire dimensions: column suffix -> name used in the results
DIMENSIONS = {
    "transcription": "transcription_quality",
    "traduction_vers_fr": "translation_caller_to_fr",
    "traduction_vers_appelant": "translation_fr_to_caller",
    "compris": "feeling_understood",
    "confiance": "confidence_real_deployment",
}
ROLES = ("appelant", "operateur")
# the operator, who only speaks French, is not asked about the translation into the caller's language
DIMENSIONS_BY_ROLE = {
    "appelant": list(DIMENSIONS),
    "operateur": [d for d in DIMENSIONS if d != "traduction_vers_appelant"],
}
SCORE_COLUMNS = [f"{role}_{d}" for role in ROLES for d in DIMENSIONS_BY_ROLE[role]]

# "Label: text" at the start of a line (label = up to 40 characters before the first colon)
_LABEL = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s*")


def _speaker(label):
    """Map a label to "Operator" / "Caller", or None if it is not a speaker label.

    Translations were labelled by target language: "<language> translation" is the
    operator's French line translated for the caller, "French translation" is the caller's line.
    """
    label = label.lower()
    if "french" in label:
        return "Caller"
    if "translation" in label or "traduction" in label:
        return "Operator"
    if label.startswith(("opérateur", "operateur", "operator")):
        return "Operator"
    if label.startswith(("appelant", "caller")):
        return "Caller"
    return None


def normalize_speakers(text):
    """Rewrite each line's speaker label as "Operator: " or "Caller: "; other lines are kept as is."""
    lines = []
    for line in text.split("\n"):
        m = _LABEL.match(line)
        speaker = m and _speaker(m.group(1))
        lines.append(f"{speaker}: {line[m.end():]}" if speaker else line)
    return "\n".join(lines)


def load_corpus():
    """Return data/corpus.csv, with the path of each call's audio file (None if missing)."""
    corpus = pd.read_csv(CORPUS_FILE, encoding="utf-8-sig", dtype={"scenario": str}, keep_default_na=False)
    for col in SCORE_COLUMNS:
        corpus[col] = pd.to_numeric(corpus[col]).astype("Int64")
    # a list keeps None (Series.map would turn it into NaN, which gr.Audio rejects)
    corpus["audio"] = [str(AUDIO_DIR / f"{c}.mp3") if (AUDIO_DIR / f"{c}.mp3").exists() else None
                       for c in corpus.call]
    return corpus


def scores(role):
    """Questionnaire scores of "appelant" or "operateur": call + the role's dimensions (named as in DIMENSIONS)."""
    corpus = load_corpus()
    dims = DIMENSIONS_BY_ROLE[role]
    df = corpus[["call"] + [f"{role}_{d}" for d in dims]]
    df = df.rename(columns={f"{role}_{d}": DIMENSIONS[d] for d in dims})
    return df.dropna(subset=[DIMENSIONS[d] for d in dims], how="all")


def save_revision(call, transcription, translation):
    """Save the manual version of a call in data/corpus.csv; return the updated corpus."""
    # re-read the file so revisions saved by another process are not overwritten
    corpus = load_corpus()
    i = corpus.index[corpus.call == call][0]
    corpus.loc[i, ["transcription_manuelle", "traduction_manuelle"]] = [
        normalize_speakers(transcription), normalize_speakers(translation)
    ]
    tmp = CORPUS_FILE.with_suffix(".tmp")
    corpus.drop(columns="audio").to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, CORPUS_FILE)
    return corpus


if __name__ == "__main__":
    c = load_corpus()
    print(c.groupby("langue").size().to_string())
    print("\nNo audio:", c.loc[c.audio.isna(), "call"].tolist())
    print("Questionnaires:", {role: int(c[[f"{role}_{d}" for d in DIMENSIONS_BY_ROLE[role]]].notna().all(axis=1).sum())
                              for role in ROLES})
