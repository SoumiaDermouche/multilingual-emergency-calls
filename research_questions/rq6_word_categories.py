"""RQ6 - Word categories: which categories of words are the most affected by transcription and translation errors?

Every word of the reference (manual transcription; manual translation) is given one category, by priority:

  emergency_services     lexicon (first: the NER model tags "Pompiers de l'Ain" as an organisation)
  named_entity           named entities (NER model: places, persons, organisations, merged because the model
                         confuses them, e.g. transliterated Arabic addresses tagged as persons), matched back onto
                         the normalised words
  number                 digits and spelled-out numbers (cardinal and ordinal, per language; "un/one/uno" excluded)
  medical_situation,     lexicons in research_questions/lexicons/rq6_lexicon.csv (patterns, * = any characters),
  instructions,          built from the corpus vocabulary, TO BE VALIDATED by the domain expert
  short_answers
  function_word          stop words
  other_content          everything else (reference category for the comparisons)

  transcription   status of each reference word in the jiwer alignment (normalised text, as in RQ1): correct,
                  substituted or deleted. Error rate = (substitutions + deletions) / words of the category.
  translation     a word of the manual translation is "missing" when it does not occur in the automatic translation
                  of the same call and speaker (multiset, normalised). Missing rate = missing / words.

Comparison with other_content: odds ratio of an error, 95% confidence interval by bootstrap over calls (the words of
a call are not independent). Per speaker, and per language for the rates.
Also: composition of the passages omitted by Whisper (runs of 5+ deleted words, RQ1) vs all caller words.

The critical information recall of RQ1 (named entities and numbers kept) is reported here, per category.

Outputs in resultats/rq6/:
  - words.csv                 one row per reference word: call, layer, speaker, language, word, category, status
  - rates.csv                 rate and odds ratio (CI) per layer x speaker x category; rates_by_language.csv
  - omitted_passages.csv      category composition of the dropped passages vs all words
  - categories.png / .pdf

Run:  python -m research_questions.rq6_word_categories
"""

import re
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path

import jiwer
import matplotlib
import numpy as np
import pandas as pd

from pre_annotation import LANGUAGES, normalize
from research_questions.common import RESULTS_DIR, SPEAKERS, split_speakers, study_corpus
from research_questions.trends import stop_words

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = RESULTS_DIR / "rq6"
LEXICON_FILE = Path(__file__).parent / "lexicons" / "rq6_lexicon.csv"
NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
REFERENCE = "other_content"
CATEGORIES = ["named_entity", "number", "emergency_services", "medical_situation", "instructions",
              "short_answers", "function_word", "other_content"]
CATEGORY_NAMES = {"named_entity": "Named entities", "number": "Numbers",
                  "emergency_services": "Emergency services", "medical_situation": "Medical / situation",
                  "instructions": "Instructions", "short_answers": "Short answers",
                  "function_word": "Function words", "other_content": "Other content words"}
LEXICON_ORDER = ["medical_situation", "instructions", "short_answers"]
SPELLED_NUMBERS = {
    "fr": "deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize vingt trente "
          "quarante cinquante soixante cent cents mille premier première deuxième troisième quatrième cinquième",
    "en": "two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen twenty "
          "thirty forty fifty sixty hundred thousand first second third fourth fifth",
    "es": "dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince veinte treinta cuarenta "
          "cincuenta cien ciento mil primero primer primera segundo tercer tercero cuarto quinto",
    "it": "due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici quindici venti trenta "
          "quaranta cinquanta cento mille primo prima secondo terzo quarto quinto",
    "ar": "اثنين اثنان اثنا اثني ثلاث ثلاثه اربع اربعه خمس خمسه سته ست سبع سبعه ثمان ثمانيه تسع تسعه عشر عشره "
          "عشرين مييه مايه الف الاول الثاني الثالث الرابع الخامس",
}
N_BOOTSTRAP = 1000
SEED = 0
RUN_MIN = 5  # dropped passage = run of at least 5 deleted reference words (as in RQ1)


def _normalize_pattern(pattern):
    """Normalise each part of a lexicon pattern like the corpus words, keeping the * wildcards."""
    return "*".join(normalize(part) for part in pattern.split("*"))


class WordCategories:
    """Error rates of transcription and translation per word category."""

    def __init__(self, device=None):
        self.device = device
        self.corpus = study_corpus()
        lex = pd.read_csv(LEXICON_FILE, encoding="utf-8-sig")
        self.lexicon = {(c, l): [_normalize_pattern(p) for p in g.pattern]
                        for (c, l), g in lex.groupby(["category", "language"])}
        self.numbers = {code: set(normalize(SPELLED_NUMBERS[code]).split()) for code in SPELLED_NUMBERS}
        self.stops = {code: stop_words(code) for code in SPELLED_NUMBERS}
        self._ner = None

    # --- labelling ----------------------------------------------------------------------

    def _entities(self, texts):
        """Normalised word sequences of the named entities (places, persons, organisations) of each text."""
        if self._ner is None:
            import torch
            from transformers import pipeline

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple",
                                 device=0 if device == "cuda" else -1)
        out = []
        for text, ents in zip(texts, self._ner(list(texts), batch_size=16)):
            spans = []
            for e in ents:
                if e["entity_group"] not in ("LOC", "PER", "ORG"):
                    continue
                start, end = e["start"], e["end"]  # widen to whole words (the tokenizer cuts words)
                while start > 0 and text[start - 1].isalnum():
                    start -= 1
                while end < len(text) and text[end].isalnum():
                    end += 1
                words = normalize(text[start:end]).split()
                if words:
                    spans.append(words)
            out.append(spans)
        return out

    def label(self, words, entities, code):
        """One category per normalised word."""
        services = self.lexicon.get(("emergency_services", code), [])
        labels = ["emergency_services" if any(fnmatchcase(w, p) for p in services) else None for w in words]
        for seq in entities:
            n = len(seq)
            for i in range(len(words) - n + 1):
                if words[i:i + n] == seq:
                    for j in range(i, i + n):
                        labels[j] = labels[j] or "named_entity"
        for i, w in enumerate(words):
            if labels[i]:
                continue
            if w.isdigit() or w in self.numbers[code]:
                labels[i] = "number"
                continue
            for cat in LEXICON_ORDER:
                if any(fnmatchcase(w, p) for p in self.lexicon.get((cat, code), [])):
                    labels[i] = cat
                    break
            if labels[i]:
                continue
            labels[i] = "function_word" if w in self.stops[code] else REFERENCE
        return labels

    # --- words and their status ---------------------------------------------------------

    def words(self):
        items = []
        for r in self.corpus.itertuples():
            tm, ta = split_speakers(r.transcription_manuelle), split_speakers(r.transcription_auto)
            mm, ma = split_speakers(r.traduction_manuelle), split_speakers(r.traduction_auto)
            for speaker in SPEAKERS:
                spoken = "fr" if speaker == "Operator" else LANGUAGES[r.langue]
                translated = LANGUAGES[r.langue] if speaker == "Operator" else "fr"
                items.append((r, speaker, "transcription", spoken, tm[speaker], ta[speaker]))
                items.append((r, speaker, "translation", translated, mm[speaker], ma[speaker]))
        print("Named entities...")
        entities = self._entities([it[4] for it in items])
        rows = []
        for (r, speaker, layer, code, ref, hyp), ents in zip(items, entities):
            ref_w = normalize(ref).split()
            if not ref_w:
                continue
            labels = self.label(ref_w, ents, code)
            status = self._status(layer, ref_w, normalize(hyp))
            for k, (w, c, s) in enumerate(zip(ref_w, labels, status)):
                rows.append({"call": r.call, "langue": r.langue, "type": r.categorie_urgence, "layer": layer,
                             "speaker": speaker, "text_language": code, "position": k, "word": w,
                             "category": c, "status": s})
        self.table = pd.DataFrame(rows)
        self.table["error"] = self.table.status != "correct"
        return self.table

    @staticmethod
    def _status(layer, ref_w, hyp):
        if layer == "transcription":
            status = ["correct"] * len(ref_w)
            for chunk in jiwer.process_words(" ".join(ref_w), hyp or "∅").alignments[0]:
                if chunk.type in ("substitute", "delete"):
                    kind = "substituted" if chunk.type == "substitute" else "deleted"
                    for j in range(chunk.ref_start_idx, chunk.ref_end_idx):
                        status[j] = kind
            return status
        available = Counter(hyp.split())
        status = []
        for w in ref_w:
            if available[w] > 0:
                available[w] -= 1
                status.append("correct")
            else:
                status.append("missing")
        return status

    # --- rates and odds ratios ----------------------------------------------------------

    @staticmethod
    def _odds_ratio(g):
        e = g.groupby("category").error.agg(["sum", "count"])
        ref = e.loc[REFERENCE] if REFERENCE in e.index else None
        rates = e["sum"] / e["count"]
        if ref is None:
            return rates, pd.Series(np.nan, index=e.index)
        odds = (e["sum"] + 0.5) / (e["count"] - e["sum"] + 0.5)  # +0.5: Haldane correction for empty cells
        return rates, odds / odds.loc[REFERENCE]

    def rates(self):
        rng = np.random.default_rng(SEED)
        rows = []
        for (layer, speaker), g in self.table.groupby(["layer", "speaker"]):
            rate, odds = self._odds_ratio(g)
            calls = g.call.unique()
            by_call = {c: x for c, x in g.groupby("call")}
            boot_rate, boot_or = [], []
            for _ in range(N_BOOTSTRAP):
                sample = pd.concat([by_call[c] for c in rng.choice(calls, len(calls))])
                br, bo = self._odds_ratio(sample)
                boot_rate.append(br)
                boot_or.append(bo)
            boot_rate, boot_or = pd.DataFrame(boot_rate), pd.DataFrame(boot_or)
            counts = g.groupby("category").agg(words=("error", "size"), errors=("error", "sum"))
            subs = g[g.status == "substituted"].groupby("category").size()
            dels = g[g.status == "deleted"].groupby("category").size()
            for cat in [c for c in CATEGORIES if c in counts.index]:
                n = counts.loc[cat, "words"]
                rows.append({"layer": layer, "speaker": speaker, "category": cat, "words": n,
                             "errors": counts.loc[cat, "errors"], "rate": rate[cat],
                             "rate_ci_low": boot_rate[cat].quantile(0.025),
                             "rate_ci_high": boot_rate[cat].quantile(0.975),
                             "substitution_rate": subs.get(cat, 0) / n if layer == "transcription" else np.nan,
                             "deletion_rate": dels.get(cat, 0) / n if layer == "transcription" else np.nan,
                             "odds_ratio": odds[cat], "or_ci_low": boot_or[cat].quantile(0.025),
                             "or_ci_high": boot_or[cat].quantile(0.975)})
        self.rates_table = pd.DataFrame(rows)
        self.rates_table["significant"] = ((self.rates_table.or_ci_low > 1) | (self.rates_table.or_ci_high < 1)) & (
            self.rates_table.category != REFERENCE)
        by_lang = self.table.groupby(["layer", "speaker", "langue", "category"]).error.agg(["size", "mean"])
        self.rates_by_language = by_lang.rename(columns={"size": "words", "mean": "rate"}).reset_index()
        return self.rates_table

    # --- dropped passages ---------------------------------------------------------------

    def omitted_passages(self):
        t = self.table[(self.table.layer == "transcription") & (self.table.speaker == "Caller")].copy()
        t["in_passage"] = False
        for _, g in t.groupby("call"):
            deleted = (g.status == "deleted").values
            idx = g.index.values
            start = None
            for k, d in enumerate(list(deleted) + [False]):
                if d and start is None:
                    start = k
                elif not d and start is not None:
                    if k - start >= RUN_MIN:
                        t.loc[idx[start:k], "in_passage"] = True
                    start = None
        share_all = t.category.value_counts(normalize=True)
        share_passage = t[t.in_passage].category.value_counts(normalize=True)
        self.passages = pd.DataFrame({"share_all_caller_words": share_all, "share_in_dropped_passages": share_passage,
                                      "words_in_dropped_passages": t[t.in_passage].category.value_counts()}
                                     ).reindex(CATEGORIES).fillna(0).rename_axis("category").reset_index()
        return self.passages

    # --- figure -------------------------------------------------------------------------

    def plot(self):
        panels = [("transcription", "Operator", "Transcription, operator (French)"),
                  ("transcription", "Caller", "Transcription, caller"),
                  ("translation", "Caller", "Translation, caller -> French"),
                  ("translation", "Operator", "Translation, operator -> caller language")]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, (layer, speaker, title) in zip(axes.flat, panels):
            g = self.rates_table[(self.rates_table.layer == layer) & (self.rates_table.speaker == speaker)]
            g = g.set_index("category").reindex([c for c in CATEGORIES if c in set(g.category)])
            y = np.arange(len(g))[::-1]
            ref_rate = g.loc[REFERENCE, "rate"] * 100
            colors = ["#b2182b" if s and r * 100 > ref_rate else "#2166ac" if s else "#999999"
                      for s, r in zip(g.significant, g.rate)]
            ax.barh(y, g.rate * 100, color=colors,
                    xerr=[(g.rate - g.rate_ci_low) * 100, (g.rate_ci_high - g.rate) * 100], capsize=2)
            ax.axvline(ref_rate, color="black", ls="--", lw=0.8)
            ax.set_yticks(y, [f"{CATEGORY_NAMES[c]} (n = {int(n)})" for c, n in zip(g.index, g.words)], fontsize=7.5)
            ax.set_title(title, fontsize=9)
        for ax in axes[1]:
            ax.set_xlabel("Words with an error (%) - 95% CI by bootstrap over calls")
        fig.suptitle("Error rate per word category (red / blue: significantly above / below other content words, "
                     "dashed line)", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"categories.{ext}", dpi=200)
        plt.close(fig)

    # --- output -------------------------------------------------------------------------

    def run(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        kw = {"index": False, "encoding": "utf-8-sig"}
        self.words()
        self.rates()
        self.omitted_passages()
        self.table.to_csv(OUT_DIR / "words.csv", **kw)
        self.rates_table.to_csv(OUT_DIR / "rates.csv", **kw)
        self.rates_by_language.to_csv(OUT_DIR / "rates_by_language.csv", **kw)
        self.passages.to_csv(OUT_DIR / "omitted_passages.csv", **kw)
        self.plot()
        print(f"-> {OUT_DIR}")
        return self


def main():
    rq = WordCategories().run()
    pd.set_option("display.width", 230, "display.max_rows", 100, "display.float_format", "{:.3f}".format)
    print(f"\n=== rates\n{rq.rates_table.to_string(index=False)}")
    print(f"\n=== dropped passages (caller)\n{rq.passages.to_string(index=False)}")


if __name__ == "__main__":
    main()
