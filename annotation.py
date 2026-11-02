"""Error annotation: taxonomy, pre-annotated errors and their validation, call by call.

pre_annotation.py writes one row per error to resultats/annotation/erreurs_pre_annotees.csv.
Each call is then checked in the review app (tab « Erreurs »); the validated table of a call is
stored in data/annotation_erreurs.json. Run this file to export the validated errors:

    python annotation.py   ->  resultats/annotation/erreurs_validees.csv, resume_par_appel.csv

Column names and category labels stay in French: they are shown to the annotators and saved in the data.
"""

import json
import os
import re
from datetime import datetime

import pandas as pd

from corpus import DATA_DIR, ROOT

ANNOTATION_DIR = ROOT / "resultats" / "annotation"
PRE_ANNOTATION = ANNOTATION_DIR / "erreurs_pre_annotees.csv"
CALLS_FILE = ANNOTATION_DIR / "appels.csv"
VALIDATIONS_FILE = DATA_DIR / "annotation_erreurs.json"

CATEGORIES = {
    "transcription": [
        "substitution phonétique", "substitution lexicale", "omission", "insertion/hallucination",
        "répétition", "code-switching", "inversion de locuteur", "entité/nombre", "forme",
    ],
    "traduction": [
        "contresens", "omission", "ajout", "terminologie", "registre", "héritée de la transcription", "forme",
    ],
}
SEVERITIES = ["critique", "majeure", "mineure"]
COLUMNS = ["id", "couche", "tour", "locuteur", "reference", "automatique", "mots_critiques",
           "categorie", "gravite", "garder"]

_SPEAKER = re.compile(r"^(Operator|Caller):\s*")


def number_turns(text):
    """Prefix each speaker turn with its number, as used in the "tour" column."""
    lines, turn = [], 0
    for line in text.split("\n"):
        if _SPEAKER.match(line):
            turn += 1
            line = f"[{turn}] {line}"
        lines.append(line)
    return "\n".join(lines)


def load_pre_annotation():
    """Pre-annotated errors as {call: DataFrame}, and the per-call information table."""
    if not PRE_ANNOTATION.exists():
        return {}, pd.DataFrame()
    df = pd.read_csv(PRE_ANNOTATION, encoding="utf-8-sig", keep_default_na=False)
    calls = pd.read_csv(CALLS_FILE, encoding="utf-8-sig")
    errors = {c: g[COLUMNS].reset_index(drop=True) for c, g in df.groupby("call", sort=False)}
    for c in calls.call:
        errors.setdefault(c, pd.DataFrame(columns=COLUMNS))
    return errors, calls


def load_validations():
    if not VALIDATIONS_FILE.exists():
        return {}
    return json.loads(VALIDATIONS_FILE.read_text(encoding="utf-8"))


def call_table(errors, validations, call):
    """Validated table of a call when saved, otherwise its pre-annotation."""
    if call in validations:
        t = pd.DataFrame(validations[call]["erreurs"], columns=COLUMNS)
    else:
        t = errors[call].copy()
    t["garder"] = t.garder.map(lambda v: str(v).strip().lower() in ("true", "1", "oui", "x"))
    return t


def clean_table(table, call):
    """Normalise a table edited in the app; returns (table, list of problems)."""
    t = table.copy()
    t = t[t.astype(str).apply(lambda r: "".join(r).strip(), axis=1) != ""]  # empty rows added by mistake
    t["garder"] = t.garder.map(lambda v: str(v).strip().lower() in ("true", "1", "oui", "x"))
    for col in ("couche", "categorie", "gravite"):
        t[col] = t[col].astype(str).str.strip().str.lower()
    # rows added in the app have no id yet
    used = set(t.id.astype(str))
    n = 1
    for i in t.index[t.id.astype(str).str.strip() == ""]:
        while f"{call}-ajout-{n:02d}" in used:
            n += 1
        t.loc[i, "id"] = f"{call}-ajout-{n:02d}"
        used.add(t.loc[i, "id"])
    problems = []
    for _, r in t[t.garder].iterrows():
        if r.couche not in CATEGORIES:
            problems.append(f"{r.id} : couche « {r.couche} » (transcription ou traduction)")
        elif r.categorie not in CATEGORIES[r.couche]:
            problems.append(f"{r.id} : catégorie « {r.categorie} » inconnue pour la {r.couche}")
        if r.gravite not in SEVERITIES:
            problems.append(f"{r.id} : gravité « {r.gravite} » (critique, majeure ou mineure)")
    return t.reset_index(drop=True), problems


def save_validation(validations, call, table, validated, comment, by=None):
    validations[call] = {
        "erreurs": table[COLUMNS].astype({"tour": str}).to_dict("records"),
        "valide": validated,
        "commentaire": comment,
        "date": datetime.now().isoformat(timespec="seconds"),
        "par": by,
    }
    # re-read the file so validations saved by another process are not overwritten
    on_disk = load_validations()
    on_disk[call] = validations[call]
    validations.update(on_disk)
    tmp = VALIDATIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(on_disk, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, VALIDATIONS_FILE)


def summarize(table):
    """Per-call summary of the kept errors: count per layer and category, worst severity, main error."""
    t = table[table.garder.map(lambda v: str(v).strip().lower() in ("true", "1", "oui", "x"))]
    if t.empty:
        return {"n_transcription": 0, "n_traduction": 0, "gravite_max": "", "erreur_principale": "", "par_categorie": {}}
    worst = next(g for g in SEVERITIES if (t.gravite == g).any()) if t.gravite.isin(SEVERITIES).any() else ""
    # main error: most frequent category among the most severe errors
    most_severe = t[t.gravite == worst] if worst else t
    main_error = most_severe.groupby(["couche", "categorie"]).size().idxmax()
    return {
        "n_transcription": int((t.couche == "transcription").sum()),
        "n_traduction": int((t.couche == "traduction").sum()),
        "gravite_max": worst,
        "erreur_principale": f"{main_error[0]} · {main_error[1]}",
        "par_categorie": t.groupby(["couche", "categorie"]).size().to_dict(),
    }


def export():
    errors, calls = load_pre_annotation()
    validations = load_validations()
    rows, summaries = [], []
    for call in calls.call:
        if not validations.get(call, {}).get("valide"):
            continue
        t = call_table(errors, validations, call)
        rows.append(t[t.garder].assign(call=call))
        s = summarize(t)
        by_category = {f"{layer}:{cat}": n for (layer, cat), n in s.pop("par_categorie").items()}
        summaries.append({"call": call, **s, **by_category})
    if not summaries:
        print("No validated call yet.")
        return
    df = pd.concat(rows, ignore_index=True)[["call"] + COLUMNS[:-1]]
    df.to_csv(ANNOTATION_DIR / "erreurs_validees.csv", index=False, encoding="utf-8-sig")
    res = calls.merge(pd.DataFrame(summaries), on="call").fillna(0)
    res.to_csv(ANNOTATION_DIR / "resume_par_appel.csv", index=False, encoding="utf-8-sig")
    print(f"{len(summaries)} validated calls, {len(df)} errors -> {ANNOTATION_DIR}")


if __name__ == "__main__":
    export()
