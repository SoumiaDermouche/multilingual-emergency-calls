"""Automatic pre-annotation of transcription and translation errors, to be checked call by call.

An error is a passage where the automatic version differs from the manual (post-edited) one,
found by a word alignment of the whole call (turn counts often differ, so turns cannot be paired).
Consecutive edits in the same turn form one error. Each error gets a first category and severity
from the rules below; categories needing judgement (registre, fine meaning shifts) are left to
the validation. Written to resultats/annotation/ (see annotation.py):

  - erreurs_pre_annotees.csv   one row per error
  - calls.csv                 per call: emergency category, duration, turns, words

Only the four study languages are kept: Turkish and Danish calls are ignored.
Column names, category and severity labels stay in French: they are shown to the annotators.
"""

import re
import unicodedata
from difflib import SequenceMatcher

import jiwer
import pandas as pd
from mutagen.mp3 import MP3

from annotation import ANNOTATION_DIR, CALLS_FILE, PRE_ANNOTATION
from corpus import AUDIO_DIR, load_corpus

LANGUAGES = {"anglais": "en", "italien": "it", "espagnol": "es", "arabe": "ar"}  # turque, danois: not studied
# a translation error is pre-labelled "inherited" when at least this share of its source turn was mistranscribed
INHERITANCE_THRESHOLD = 0.3
EMERGENCY_CATEGORIES = {"1": "incendie", "2": "secours à la personne", "3": "accident"}

_SPEAKER = re.compile(r"^(Operator|Caller):\s*")
_ARABIC = re.compile(r"[؀-ۿ]")

# Arabic: short vowels / shadda / sukun, tatweel, and letter variants normalised as is usual for Arabic ASR
_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_ARABIC_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# word lists on normalised text (lowercase, no punctuation, Arabic letters unified: ة->ه, أ/إ/آ->ا, ى->ي)
FUNCTION_WORDS = {
    "fr": set("le la les l un une des de du d et ou à au aux en dans sur pour par avec ce c ça cette ces est "
              "il elle ils on je j vous nous tu me m te se s y a ai as ont qui que qu ne n pas oui non mais "
              "donc alors bon ok euh ah bien très voilà".split()),
    "en": set("the a an and or of to in on at for with is are was were be it it's this that i you he she we "
              "they my your his her its me him them do does did not no yes so ok okay uh um ah oh there what s".split()),
    "it": set("il lo la i gli le l un una uno di d da in con su per e o è sono che non sì si mi ti ci vi ho ha "
              "hanno del della dei al alla nel nella ok eh ah allora bene".split()),
    "es": set("el la los las un una unos de del a al en con por para y o es son que no sí si me te se le lo "
              "hay mi su ok eh ah pues bueno".split()),
    "ar": set("في من علي الي ان و او لا نعم هل ما هذا هذه هناك انا انت هو هي نحن عن مع قد كان اه يا ثم لكن "
              "التي الذي ذلك".split()),
}
NUMBER_WORDS = set(
    "zéro deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize vingt trente "
    "quarante cinquante soixante cent mille premier première deuxième troisième quatrième cinquième "
    "zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred "
    "thousand first third fourth fifth "
    "due tre quattro cinque sei sette otto nove dieci venti trenta cento primo prima terzo quarto quinto "
    "cero dos tres cuatro cinco seis siete ocho nueve diez veinte treinta cien ciento mil primero tercero cuarto quinto "
    "واحد اثنين اثنان ثلاث ثلاثه اربع اربعه خمس خمسه سته سبع سبعه ثمانيه تسع تسعه عشر عشره عشرين مئه الف "
    "الاول الثاني الثالث الرابع الخامس".split()
)
# emergency vocabulary (stems): a word starting with one of them (containing it, in Arabic) is critical
EMERGENCY_STEMS = (
    "feu incend brûl bless sang respir conscien malais étage adress rue avenue chemin accident voiture "
    "victim enfant bébé fumée douleur cœur coeur tomb chute "
    "fire burn injur blood breath conscious unconscious faint floor address street road car child baby "
    "smoke pain heart fell fall "
    "fuoco brucia ferit sangue cosci sven piano indirizz via strada incident macchina auto bambin fumo "
    "dolore cuore cadut "
    "fuego quema herid sangre consci desmay piso direcc calle carretera coche vehícul niñ humo dolor "
    "corazón caíd"
).split()
EMERGENCY_STEMS_AR = "حريق نار حرق مصاب جرح دم تنفس واعي وعي اغم طابق عنوان شارع طريق حادث سياره ضحي طفل دخان الم قلب سقط".split()


# --- words ----------------------------------------------------------------------------

def normalize(text):
    """Lowercase, Arabic normalisation, punctuation removed (apostrophes split words: "l'adresse" -> "l adresse")."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _ARABIC_DIACRITICS.sub("", text).translate(_ARABIC_LETTERS).translate(_ARABIC_DIGITS)
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split())


def words(text):
    """Normalised words of a text, each with its turn number (1-based) and speaker."""
    out, turn, speaker = [], 0, None
    for line in text.split("\n"):
        m = _SPEAKER.match(line)
        if m:
            speaker = m.group(1)
            turn += 1
            line = line[m.end():]
        # "[non transcrit]": the system produced nothing for this turn -> its words become omissions
        for w in normalize(line.replace("[non transcrit]", " ")).split():
            out.append((w, turn, speaker))
    return out


def count_turns(text):
    return sum(bool(_SPEAKER.match(l)) for l in text.split("\n"))


def is_critical(w, entities):
    if w.isdigit() or w in NUMBER_WORDS or w in entities:
        return True
    if _ARABIC.search(w):
        return any(s in w for s in EMERGENCY_STEMS_AR)
    return w.startswith(tuple(EMERGENCY_STEMS))


def is_function_word(w):
    return any(w in f for f in FUNCTION_WORDS.values())


def language_of(ws, expected_langs):
    """Language of a span among the two expected ones (French and the call language), None if unclear."""
    if not ws:
        return None
    if "ar" in expected_langs:
        arabic = sum(bool(_ARABIC.search(w)) for w in ws)
        return "ar" if arabic > len(ws) / 2 else "fr" if arabic == 0 else None
    counts = {l: sum(w in FUNCTION_WORDS[l] for w in ws) for l in expected_langs}
    best = max(counts, key=counts.get)
    # a tie (or no function word) is undecidable: short spans rarely contain function words
    return best if counts[best] > 0 and list(counts.values()).count(counts[best]) == 1 else None


def severity(ws, entities):
    if any(is_critical(w, entities) for w in ws):
        return "critique"
    if all(is_function_word(w) for w in ws):
        return "mineure"
    return "majeure"


# --- alignment ------------------------------------------------------------------------

def align(ref, hyp):
    """Errors between two word lists, as dicts (ref/hyp word ranges, operations, turn, speaker)."""
    if not ref:
        return [], []
    r = " ".join(w for w, _, _ in ref)
    h = " ".join(w for w, _, _ in hyp) or "∅"
    chunks = jiwer.process_words(r, h).alignments[0]

    errors, current = [], None
    for m in chunks:
        if m.type == "equal":
            current = None
            continue
        # an insertion belongs to the turn of the reference word just before it
        i_ref = m.ref_start_idx if m.type != "insert" else max(m.ref_start_idx - 1, 0)
        turn, speaker = ref[i_ref][1], ref[i_ref][2]
        if m.type == "insert" and hyp:
            speaker = hyp[m.hyp_start_idx][2] or speaker
        op = {"substitute": "S", "delete": "D", "insert": "I"}[m.type]
        if current and current["tour"] == turn and current["r1"] == m.ref_start_idx and current["h1"] == m.hyp_start_idx:
            current.update(r1=m.ref_end_idx, h1=m.hyp_end_idx)
            current["ops"].add(op)
        else:
            current = {"tour": turn, "locuteur": speaker, "r0": m.ref_start_idx, "r1": m.ref_end_idx,
                       "h0": m.hyp_start_idx, "h1": m.hyp_end_idx, "ops": {op}}
            errors.append(current)
    if h == "∅":
        for e in errors:
            e["h0"] = e["h1"] = 0
    return errors, chunks


def speaker_swaps(ref, hyp, chunks):
    """Runs of >= 3 correctly recognised words attributed to the wrong speaker."""
    runs, current = [], None
    for m in chunks:
        if m.type != "equal":
            current = None
            continue
        for k in range(m.ref_end_idx - m.ref_start_idx):
            i, j = m.ref_start_idx + k, m.hyp_start_idx + k
            if hyp[j][2] and ref[i][2] and hyp[j][2] != ref[i][2]:
                if current and current["r1"] == i and current["tour"] == ref[i][1]:
                    current["r1"], current["h1"] = i + 1, j + 1
                else:
                    current = {"tour": ref[i][1], "locuteur": ref[i][2], "r0": i, "r1": i + 1, "h0": j, "h1": j + 1,
                               "ops": {"R"}}
                    runs.append(current)
            else:
                current = None
        current = None
    return [r for r in runs if r["r1"] - r["r0"] >= 3]


def pair_moves(errors, ref, hyp):
    """Merge an omission and an insertion of (nearly) the same words: the passage was recognised but
    placed in another turn, which the whole-call alignment sees as two errors."""
    ref_ws, hyp_ws = [w for w, _, _ in ref], [w for w, _, _ in hyp]
    omissions = [e for e in errors if e["ops"] == {"D"}]
    insertions = [e for e in errors if e["ops"] == {"I"}]
    for o in omissions:
        text = " ".join(ref_ws[o["r0"]:o["r1"]])
        candidates = [(SequenceMatcher(None, text, " ".join(hyp_ws[i["h0"]:i["h1"]])).ratio(), k)
                      for k, i in enumerate(insertions) if abs(i["tour"] - o["tour"]) <= 2]
        if not candidates or max(candidates)[0] < 0.75:
            continue
        ratio, k = max(candidates)
        i = insertions.pop(k)
        errors.remove(i)
        o["h0"], o["h1"] = i["h0"], i["h1"]
        # identical words: only the place changed (turn split, order or speaker); otherwise also misrecognised
        o["ops"] = {"M"} if ratio == 1 else {"S", "M"}
        o["auto_speaker"] = hyp[i["h0"]][2]
    return errors


def is_repetition(hyp_ws, h0, h1):
    """Inserted words that repeat the words just before or after them, or a word looping >= 3 times."""
    ins = hyp_ws[h0:h1]
    n = len(ins)
    if hyp_ws[max(0, h0 - n):h0] == ins or hyp_ws[h1:h1 + n] == ins:
        return True
    return any(ins[k] == ins[k + 1] == ins[k + 2] for k in range(n - 2))


# --- classification -------------------------------------------------------------------

def classify_transcription(e, ref_ws, hyp_ws, entities, code):
    r, h = ref_ws[e["r0"]:e["r1"]], hyp_ws[e["h0"]:e["h1"]]
    expected = "fr" if e["locuteur"] == "Operator" else code
    expected_langs = {"fr", code}
    lr, lh = language_of(r, expected_langs), language_of(h, expected_langs)
    if e["ops"] in ({"R"}, {"M"}):
        return "inversion de locuteur"
    if e["ops"] == {"I"}:
        return "répétition" if is_repetition(hyp_ws, e["h0"], e["h1"]) else "insertion/hallucination"
    # the speaker switched language (e.g. a caller answering in French), or the system wrote another language
    if (lr and lr != expected) or (lr and lh and lr != lh):
        return "code-switching"
    if e["ops"] == {"D"}:
        return "omission"
    if any(w.isdigit() or w in NUMBER_WORDS or w in entities for w in r + h):
        return "entité/nombre"
    if all(is_function_word(w) for w in r + h):
        return "forme"
    return "substitution phonétique" if SequenceMatcher(None, " ".join(r), " ".join(h)).ratio() >= 0.5 \
        else "substitution lexicale"


def classify_translation(e, ref_ws, hyp_ws, inherited_turns):
    r, h = ref_ws[e["r0"]:e["r1"]], hyp_ws[e["h0"]:e["h1"]]
    # passages displaced to another turn follow the segmentation of the automatic transcription
    if e["ops"] in ({"R"}, {"M"}):
        return "héritée de la transcription"
    # the source turn was already wrongly transcribed: the translation error most likely comes from it
    if e["tour"] in inherited_turns:
        return "héritée de la transcription"
    if e["ops"] == {"D"}:
        return "omission"
    if e["ops"] == {"I"}:
        return "ajout"
    if all(is_function_word(w) for w in r + h):
        return "forme"
    content_r = [w for w in r if not is_function_word(w)]
    content_h = [w for w in h if not is_function_word(w)]
    return "terminologie" if len(content_r) == len(content_h) == 1 else "contresens"


def entities_per_call(ner, texts):
    """Words of the named entities (places, persons, organisations) found in each text."""
    outputs = ner([t.replace("Operator:", "").replace("Caller:", "") for t in texts], batch_size=8)
    return [{w for e in ents if e["entity_group"] in ("LOC", "PER", "ORG")
             for w in normalize(e["word"]).split() if len(w) > 1}
            for ents in outputs]


def error_severity(e, r, h, entities):
    if e["ops"] == {"M"}:
        # words intact, only displaced: serious only if they end up on the other speaker
        return "majeure" if e["auto_speaker"] != e["locuteur"] else "mineure"
    if e["ops"] == {"R"}:
        return "critique" if any(is_critical(w, entities) for w in r) else "majeure"
    return severity(r + h, entities)


def error_rows(call, layer, ref, hyp, errors, classify, entities):
    ref_ws, hyp_ws = [w for w, _, _ in ref], [w for w, _, _ in hyp]
    rows = []
    for k, e in enumerate(sorted(errors, key=lambda e: (e["r0"], e["h0"])), 1):
        r, h = ref_ws[e["r0"]:e["r1"]], hyp_ws[e["h0"]:e["h1"]]
        if e["ops"] == {"R"}:
            other = "Caller" if e["locuteur"] == "Operator" else "Operator"
            h = h + [f"(attribué à {other})"]
        elif "M" in e["ops"]:
            h = h + ["(déplacé" + (f", attribué à {e['auto_speaker']})" if e["auto_speaker"] != e["locuteur"] else ")")]
        rows.append({
            "call": call, "id": f"{call}-{layer[:5]}-{k:03d}", "couche": layer, "tour": e["tour"],
            "locuteur": e["locuteur"] or "", "operations": "".join(sorted(e["ops"])),
            "reference": " ".join(r), "automatique": " ".join(h),
            "mots_critiques": " ".join(w for w in dict.fromkeys(r + h) if is_critical(w, entities)),
            "categorie": classify(e), "gravite": error_severity(e, r, h, entities),
            "garder": True,
        })
    return rows


def duration(call):
    path = AUDIO_DIR / f"{call}.mp3"
    return round(MP3(path).info.length, 1) if path.exists() else None


def main():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1

    df = load_corpus()
    df = df[df.langue.isin(LANGUAGES)]
    df = df[(df.transcription_auto != "") & (df.transcription_manuelle != "")].reset_index(drop=True)

    print("Named entities (NER)...")
    ner = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple",
                   device=device)
    ent_transcription = entities_per_call(ner, list(df.transcription_manuelle))
    ent_translation = entities_per_call(ner, list(df.traduction_manuelle))

    rows, calls = [], []
    for row, et, ed in zip(df.itertuples(), ent_transcription, ent_translation):
        code = LANGUAGES[row.langue]

        ref, hyp = words(row.transcription_manuelle), words(row.transcription_auto)
        err, chunks = align(ref, hyp)
        err = pair_moves(err, ref, hyp) + speaker_swaps(ref, hyp, chunks)
        ref_ws, hyp_ws = [w for w, _, _ in ref], [w for w, _, _ in hyp]
        transcription_rows = error_rows(row.call, "transcription", ref, hyp, err,
                                        lambda e: classify_transcription(e, ref_ws, hyp_ws, et, code), et)

        # share of each source turn that was wrongly transcribed (non-minor errors); manual
        # transcription and translation turns correspond one to one when their counts match
        turn_length = pd.Series([t for _, t, _ in ref]).value_counts()
        faulty = {}
        for e, l in zip(sorted(err, key=lambda e: (e["r0"], e["h0"])), transcription_rows):
            if l["gravite"] != "mineure":
                faulty[e["tour"]] = faulty.get(e["tour"], 0) + max(e["r1"] - e["r0"], e["h1"] - e["h0"])
        faulty_share = {t: min(1, n / turn_length.get(t, 1)) for t, n in faulty.items()}
        inherited_turns = set()
        if count_turns(row.transcription_manuelle) == count_turns(row.traduction_manuelle):
            inherited_turns = {t for t, p in faulty_share.items() if p >= INHERITANCE_THRESHOLD}
        ref_t, hyp_t = words(row.traduction_manuelle), words(row.traduction_auto)
        err_t, chunks_t = align(ref_t, hyp_t)
        err_t = pair_moves(err_t, ref_t, hyp_t) + speaker_swaps(ref_t, hyp_t, chunks_t)
        rt_ws, ht_ws = [w for w, _, _ in ref_t], [w for w, _, _ in hyp_t]
        translation_rows = error_rows(row.call, "traduction", ref_t, hyp_t, err_t,
                                      lambda e: classify_translation(e, rt_ws, ht_ws, inherited_turns), ed)
        for l in translation_rows:
            l["part_source_fautive"] = round(faulty_share.get(l["tour"], 0), 2)
        rows += transcription_rows + translation_rows

        turns = count_turns(row.transcription_manuelle)
        d = duration(row.call)
        scenario = row.call.split("_", 1)[1]
        calls.append({
            "call": row.call, "langue": row.langue, "scenario": scenario,
            "categorie_urgence": EMERGENCY_CATEGORIES.get(scenario.split(".")[0], ""),
            "duree_s": d, "tours_ref": turns, "tours_auto": count_turns(row.transcription_auto),
            "mots_ref": len(ref), "mots_par_tour": round(len(ref) / turns, 1) if turns else None,
            # no per-turn timestamps: mean turn duration is estimated as call duration / turns
            "secondes_par_tour": round(d / turns, 1) if d and turns else None,
            "pre_erreurs_transcription": len(transcription_rows), "pre_erreurs_traduction": len(translation_rows),
        })

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    errors = pd.DataFrame(rows)
    errors.to_csv(PRE_ANNOTATION, index=False, encoding="utf-8-sig")
    calls = pd.DataFrame(calls)
    calls.to_csv(CALLS_FILE, index=False, encoding="utf-8-sig")

    print(f"\n{len(calls)} calls, {len(errors)} pre-annotated errors -> {ANNOTATION_DIR}")
    print(errors.groupby(["couche", "categorie"]).size().to_string())
    print(errors.groupby(["couche", "gravite"]).size().to_string())
    print(calls.groupby("langue")[["duree_s", "tours_ref", "mots_par_tour"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
