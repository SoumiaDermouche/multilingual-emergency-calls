"""Corpus loading: links each audio file to its transcriptions and translations.

The original spreadsheets are never modified. Corrections made in the review
interface are stored in data/revisions.json.

Speaker labels are normalised on load: every line starts with "Operator:" or
"Caller:", in both transcriptions and translations, for every language.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from odf import opendocument, teletype
from odf.table import Table, TableCell, TableRow
from odf.text import P

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
TABLES_DIR = DATA_DIR / "translation+ transcription"
REVISIONS_FILE = DATA_DIR / "revisions.json"

COLONNES = ["transcription", "transcription_corrigee", "traduction", "traduction_corrigee"]

# language -> (audio file prefix, spreadsheet, sheet)
SOURCES = {
    "anglais": ("anglais", "iappel_db_english.xlsx", 0),
    "italien": ("italien", "iappel_db_italian.xlsx", "après correction"),
    "espagnol": ("espagnol", "iappel_db_spanish.ods", 0),
    # automatic transcription and translation only; corrected versions are in data/revisions.json
    "arabe": ("arabe", "iappel_db_arabic.xlsx", 0),
}

# "Label: text" at the start of a line (label = up to 40 characters before the first colon)
_ETIQUETTE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s*")


def _role(etiquette):
    """Map a spreadsheet label to "Operator" / "Caller", or None if it is not a speaker label.

    Translations were labelled by target language: "<language> translation" is the
    operator's French line translated for the caller, "French translation" is the caller's line.
    """
    e = etiquette.lower()
    if "french" in e:
        return "Caller"
    if "translation" in e or "traduction" in e:
        return "Operator"
    if e.startswith(("opérateur", "operateur", "operator")):
        return "Operator"
    if e.startswith(("appelant", "caller")):
        return "Caller"
    return None


def normaliser_roles(texte):
    """Rewrite each line's speaker label as "Operator: " or "Caller: "; other lines are kept as is."""
    lignes = []
    for ligne in texte.split("\n"):
        m = _ETIQUETTE.match(ligne)
        role = m and _role(m.group(1))
        lignes.append(f"{role}: {ligne[m.end():]}" if role else ligne)
    return "\n".join(lignes)


def _lire_ods(chemin):
    """Read the first sheet of an .ods file, keeping line breaks.

    pandas joins the paragraphs of an ODS cell without a separator
    ("écoute.Appelant:"), hence this dedicated reader.
    """
    doc = opendocument.load(chemin)
    feuille = doc.spreadsheet.getElementsByType(Table)[0]
    lignes = []
    for row in feuille.getElementsByType(TableRow):
        valeurs = []
        for cell in row.getElementsByType(TableCell):
            repet = int(cell.getAttribute("numbercolumnsrepeated") or 1)
            texte = "\n".join(teletype.extractText(p) for p in cell.getElementsByType(P))
            valeurs.extend([texte] * min(repet, 50))
        if any(v.strip() for v in valeurs):
            lignes.append(valeurs)
    entete = lignes[0]
    n = len([c for c in entete if c.strip()])
    df = pd.DataFrame([l[:n] + [""] * (n - len(l)) for l in lignes[1:]], columns=entete[:n])
    return df.replace("", None)


def _lire_tableur(fichier, feuille):
    chemin = TABLES_DIR / fichier
    if chemin.suffix == ".ods":
        return _lire_ods(chemin)
    # dtype=str keeps "2.10" distinct from "2.1"
    return pd.read_excel(chemin, sheet_name=feuille, dtype=str)


def charger_corpus():
    """Return a DataFrame with one row per call (id, language, scenario, audio, texts)."""
    frames = []
    for langue, (prefixe, fichier, feuille) in SOURCES.items():
        if not (TABLES_DIR / fichier).exists():
            continue
        df = _lire_tableur(fichier, feuille)
        df["numero_scenario"] = df["numero_scenario"].astype(str).str.strip()
        df["langue"] = langue
        df["id"] = prefixe + "_" + df["numero_scenario"]
        df["audio"] = df["id"].map(
            lambda i: str(AUDIO_DIR / f"{i}.mp3") if (AUDIO_DIR / f"{i}.mp3").exists() else None
        )
        frames.append(df)
    corpus = pd.concat(frames, ignore_index=True)
    for col in COLONNES:
        corpus[col] = corpus[col].fillna("").astype(str).str.strip().map(normaliser_roles)
    return corpus[["id", "langue", "numero_scenario", "audio"] + COLONNES]


def charger_revisions():
    if not REVISIONS_FILE.exists():
        return {}
    revisions = json.loads(REVISIONS_FILE.read_text(encoding="utf-8"))
    for r in revisions.values():
        r["transcription"] = normaliser_roles(r["transcription"])
        r["traduction"] = normaliser_roles(r["traduction"])
    return revisions


def enregistrer_revision(revisions, id_appel, transcription, traduction, valide, par=None):
    revisions[id_appel] = {
        "transcription": normaliser_roles(transcription),
        "traduction": normaliser_roles(traduction),
        "valide": valide,
        "date": datetime.now().isoformat(timespec="seconds"),
        "par": par,
    }
    # re-read the file so revisions saved by another process are not overwritten
    sur_disque = charger_revisions()
    sur_disque[id_appel] = revisions[id_appel]
    revisions.update(sur_disque)
    tmp = REVISIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(sur_disque, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, REVISIONS_FILE)


if __name__ == "__main__":
    c = charger_corpus()
    print(c.groupby("langue").size().to_string())
    print("\nNo audio:", c.loc[c.audio.isna(), "id"].tolist())
    print("No corrected transcription:", c.loc[c.transcription_corrigee == "", "id"].tolist())
