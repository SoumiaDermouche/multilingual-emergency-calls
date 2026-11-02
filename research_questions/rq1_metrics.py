"""RQ1 - Transcription and translation quality metrics, and correlations between metrics.

Each automatic text is compared with its manual (post-edited) version, per call and per speaker
(Operator = French, Caller = call language; "all" = the whole call):

  transcription (Whisper)   WER, CER, MER, WIL, substitution / deletion / insertion rates (normalised text),
                            SemDist, BERTScore F1, critical information recall, turn count difference
  translation (LLM)         BLEU, chrF++, TER (raw text), SemDist, BERTScore F1,
                            critical information recall

Translation metrics are end to end: they include the transcription errors propagated to the translation.

Outputs in resultats/rq1/:
  - metrics_per_call.csv        one row per call x layer x speaker
  - metrics_per_language.csv    per layer x speaker x language (+ TOTAL); error rates and BLEU/chrF/TER
                                are micro-averaged (pooled counts), the other metrics are means over calls
  - omissions_per_language.csv  deleted words by length of the deletion run (isolated words vs dropped passages)
  - correlations.csv            Spearman rho between all metrics at call level (whole call), Holm-corrected
  - correlations_by_language.csv
  - excluded_rows.csv           only if some translation rows are left out: the manual "translation" is a
                                copy of the manual transcription (never translated), see _drop_untranslated_references
  - correlations.png / .pdf

Run:  python -m research_questions.rq1_metrics
"""

import re
from difflib import SequenceMatcher

import jiwer
import matplotlib
import numpy as np
import pandas as pd
import sacrebleu

from research_questions.common import (ALL, RESULTS_DIR, SPEAKERS, count_turns, normalize, spearman_matrix,
                                       split_speakers, stars, study_corpus, text_language)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT_DIR = RESULTS_DIR / "rq1"
LAYERS = ("transcription", "translation")

SEMDIST_MODEL = "BAAI/bge-m3"
BERTSCORE_MODEL = "xlm-roberta-large"
NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"

# call-level metrics used for the correlations; prefix asr_ = transcription, mt_ = translation.
# Critical information recall (info_recall) is still computed but left out: it is analysed with the word
# categories in RQ6.
CORRELATED = {
    "transcription": ["wer", "cer", "sub_rate", "del_rate", "ins_rate", "semdist", "bertscore_f1", "turn_diff"],
    "translation": ["bleu", "chrf", "ter", "semdist", "bertscore_f1"],
}
PREFIX = {"transcription": "asr", "translation": "mt"}
# a manual translation this similar to its source was never translated (see _drop_untranslated_references)
UNTRANSLATED_SIMILARITY = 0.95

_NUMBER = re.compile(r"\d+")
# deletion runs (consecutive deleted reference words) by length: isolated words vs whole dropped passages
DELETION_RUNS = {"1-2": (1, 2), "3-4": (3, 4), "5+": (5, None)}


class QualityMetrics:
    """Compute, summarise and correlate the quality metrics of RQ1."""

    def __init__(self, device=None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.corpus = study_corpus()
        self.pairs = self._build_pairs()
        self.per_call = None
        self.per_language = None
        self.correlations = None
        self.correlations_by_language = None

    # --- data -------------------------------------------------------------------------

    def _build_pairs(self):
        """One row per call x layer x speaker: reference (manual), hypothesis (automatic), source."""
        rows = []
        for r in self.corpus.itertuples():
            texts = {
                "transcription": (split_speakers(r.transcription_manuelle), split_speakers(r.transcription_auto)),
                "translation": (split_speakers(r.traduction_manuelle), split_speakers(r.traduction_auto)),
            }
            source = texts["transcription"][0]
            for layer, (ref, hyp) in texts.items():
                for speaker in SPEAKERS + (ALL,):
                    rows.append({
                        "call": r.call, "langue": r.langue, "scenario": r.scenario,
                        "categorie_urgence": r.categorie_urgence, "layer": layer, "speaker": speaker,
                        "text_language": text_language(layer, speaker, r.langue),
                        "ref": ref[speaker], "hyp": hyp[speaker],
                        "src": source[speaker] if layer == "translation" else "",
                        "turns_ref": count_turns(r.transcription_manuelle) if speaker == ALL else None,
                        "turns_auto": count_turns(r.transcription_auto) if speaker == ALL else None,
                    })
        pairs = pd.DataFrame(rows)
        # a speaker with no manual text cannot be evaluated
        pairs = pairs[pairs.ref != ""].reset_index(drop=True)
        return self._drop_untranslated_references(pairs)

    def _drop_untranslated_references(self, pairs):
        """Drop translation rows whose manual "translation" is a copy of the manual transcription (the
        reference was never translated: e.g. the Italian caller turns). The whole-call row of such a call
        is dropped too, since it contains that speaker. The dropped rows are kept in self.excluded."""
        tr = pairs.layer == "translation"
        copied = tr & (pairs.speaker != ALL) & pd.Series(
            [SequenceMatcher(None, r, s).ratio() >= UNTRANSLATED_SIMILARITY if t else False
             for r, s, t in zip(pairs.ref, pairs.src, tr)], index=pairs.index)
        calls = set(pairs.loc[copied, "call"])
        drop = copied | (tr & (pairs.speaker == ALL) & pairs.call.isin(calls))
        self.excluded = pairs.loc[drop, ["call", "langue", "layer", "speaker"]].assign(
            reason="manual translation = copy of the manual transcription")
        if drop.any():
            summary = self.excluded.groupby(["langue", "speaker"]).size().to_dict()
            print(f"Excluded translation rows (untranslated reference): {summary}")
        return pairs[~drop].reset_index(drop=True)

    # --- metrics ----------------------------------------------------------------------

    @staticmethod
    def _error_rates(ref, hyp):
        """jiwer word and character counts on normalised text (an empty hypothesis = one placeholder word)."""
        ref, hyp = normalize(ref), normalize(hyp) or "∅"
        w = jiwer.process_words(ref, hyp)
        c = jiwer.process_characters(ref, hyp)
        n = w.hits + w.substitutions + w.deletions
        n_chars = c.hits + c.substitutions + c.deletions
        return {
            "wer": w.wer, "cer": c.cer, "mer": w.mer, "wil": w.wil,
            "sub_rate": w.substitutions / n, "del_rate": w.deletions / n, "ins_rate": w.insertions / n,
            "ref_words": n, "hits": w.hits, "substitutions": w.substitutions, "deletions": w.deletions,
            "insertions": w.insertions, "ref_chars": n_chars,
            "char_errors": c.substitutions + c.deletions + c.insertions,
        }

    @staticmethod
    def _translation_scores(ref, hyp):
        return {
            "bleu": sacrebleu.sentence_bleu(hyp, [ref]).score,
            "chrf": sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score,
            "ter": sacrebleu.sentence_ter(hyp, [ref]).score,
        }

    def _surface(self):
        rows = []
        for p in self.pairs.itertuples():
            if p.layer == "transcription":
                m = self._error_rates(p.ref, p.hyp)
                if p.speaker == ALL:
                    m["turn_diff"] = abs(p.turns_auto - p.turns_ref)
            else:
                m = self._translation_scores(p.ref, p.hyp)
            rows.append(m)
        return pd.DataFrame(rows, index=self.pairs.index)

    def _semdist(self):
        """SemDist (Kim et al., 2021): 1 - cosine similarity of sentence embeddings."""
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(SEMDIST_MODEL, device=self.device)
        e_ref = enc.encode(list(self.pairs.ref), normalize_embeddings=True, convert_to_tensor=True, batch_size=16)
        e_hyp = enc.encode(list(self.pairs.hyp), normalize_embeddings=True, convert_to_tensor=True, batch_size=16)
        return pd.Series((1 - (e_ref * e_hyp).sum(dim=1)).cpu().numpy(), index=self.pairs.index)

    def _bertscore(self):
        """BERTScore F1 without baseline rescaling (calls mix two languages)."""
        from bert_score import score

        # an empty hypothesis (nothing produced) gets the same placeholder as in _error_rates:
        # bert-score's empty-string path is broken with transformers 5
        hyps = [h.strip() or "∅" for h in self.pairs.hyp]
        _, _, f1 = score(hyps, list(self.pairs.ref), model_type=BERTSCORE_MODEL,
                         device=self.device, batch_size=16)
        return pd.Series(f1.numpy(), index=self.pairs.index)

    def _critical_information(self):
        """Named entities (places, people, organisations) and numbers of the reference still present in the
        hypothesis: (count in the reference, count kept, missing items)."""
        from transformers import pipeline

        ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple",
                       device=0 if self.device == "cuda" else -1)
        refs = list(self.pairs.ref)
        items = [self._information_items(t, ents) for t, ents in zip(refs, ner(refs, batch_size=16))]
        hyps = [normalize(h) for h in self.pairs.hyp]
        kept = [{i for i in it if self._present(i, h)} for it, h in zip(items, hyps)]
        return pd.DataFrame({
            "info_ref": [len(i) for i in items],
            "info_kept": [len(k) for k in kept],
            "info_missing": [", ".join(sorted(i - k)) for i, k in zip(items, kept)],
        }, index=self.pairs.index)

    @staticmethod
    def _information_items(text, entities):
        # spans taken from the original text and widened to whole words: the tokenizer's
        # "word" field cuts words into pieces ("pomp", "iers") or glues them together
        spans = []
        for e in entities:
            if e["entity_group"] not in ("LOC", "PER", "ORG"):
                continue
            start, end = e["start"], e["end"]
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            if spans and start <= spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))
        items = {normalize(text[s:e]) for s, e in spans}
        items |= set(_NUMBER.findall(normalize(text)))
        return {i for i in items if len(i) > 1 or i.isdigit()}

    @staticmethod
    def _present(item, hyp):
        """True if item occurs in hyp, allowing small spelling differences (similarity >= 0.8)."""
        if item.isdigit():
            return item in hyp.split()
        if item in hyp:
            return True
        n = len(item)
        return any(SequenceMatcher(None, item, hyp[i:i + n]).ratio() >= 0.8
                   for i in range(0, max(1, len(hyp) - n + 1)))

    def compute(self):
        print("Surface metrics (jiwer, sacrebleu)...")
        m = self._surface()
        print("SemDist...")
        m["semdist"] = self._semdist()
        print("BERTScore...")
        m["bertscore_f1"] = self._bertscore()
        print("Critical information (NER)...")
        m = m.join(self._critical_information())
        m["info_recall"] = m.info_kept / m.info_ref.where(m.info_ref > 0)
        self.per_call = self.pairs.drop(columns=["ref", "hyp", "src"]).join(m)
        return self.per_call

    def deletion_runs(self):
        """Omissions by length of the run of consecutive deleted reference words, per speaker and language:
        short runs are isolated missed words, long runs (5+) are whole passages Whisper did not transcribe."""
        rows = []
        tr = self.pairs[(self.pairs.layer == "transcription") & (self.pairs.speaker != ALL)]
        for p in tr.itertuples():
            out = jiwer.process_words(normalize(p.ref), normalize(p.hyp) or "∅")
            for chunk in out.alignments[0]:
                if chunk.type == "delete":
                    rows.append({"call": p.call, "langue": p.langue, "speaker": p.speaker,
                                 "length": chunk.ref_end_idx - chunk.ref_start_idx})
        runs = pd.DataFrame(rows)
        table = []
        for (speaker, language), g in list(runs.groupby(["speaker", "langue"])) + [
                ((s, "TOTAL"), g) for s, g in runs.groupby("speaker")]:
            row = {"speaker": speaker, "langue": language, "deleted_words": g.length.sum()}
            for name, (lo, hi) in DELETION_RUNS.items():
                sel = g[(g.length >= lo) & (g.length <= (hi or g.length.max()))]
                row[f"words_in_runs_{name}"] = sel.length.sum()
                row[f"share_runs_{name}"] = sel.length.sum() / g.length.sum()
                row[f"n_runs_{name}"] = len(sel)
            row["calls_with_run_5+"] = g[g.length >= 5].call.nunique()
            row["longest_run"] = g.length.max()
            table.append(row)
        self.omissions = pd.DataFrame(table)
        return self.omissions

    # --- summaries --------------------------------------------------------------------

    def _pooled(self, g):
        """Metrics of a group of calls: pooled counts for error rates and BLEU/chrF/TER, means for the rest."""
        layer = g.layer.iloc[0]
        out = {"calls": g.call.nunique()}
        if layer == "transcription":
            n = g.ref_words.sum()
            out |= {
                "wer": (g.substitutions + g.deletions + g.insertions).sum() / n,
                "cer": g.char_errors.sum() / g.ref_chars.sum(),
                "sub_rate": g.substitutions.sum() / n, "del_rate": g.deletions.sum() / n,
                "ins_rate": g.insertions.sum() / n, "ref_words": n,
                "wer_mean_per_call": g.wer.mean(),
            }
            if "turn_diff" in g and g.turn_diff.notna().any():
                out["turn_diff"] = g.turn_diff.mean()
        else:
            pairs = self.pairs.loc[g.index]
            refs, hyps = [list(pairs.ref)], list(pairs.hyp)
            out |= {
                "bleu": sacrebleu.corpus_bleu(hyps, refs).score,
                "chrf": sacrebleu.corpus_chrf(hyps, refs, word_order=2).score,
                "ter": sacrebleu.corpus_ter(hyps, refs).score,
            }
        out |= {"semdist": g.semdist.mean(), "bertscore_f1": g.bertscore_f1.mean(),
                "info_recall": g.info_kept.sum() / g.info_ref.sum()}
        return out

    def summarize(self):
        rows = []
        for (layer, speaker), g in self.per_call.groupby(["layer", "speaker"], sort=False):
            for language, gl in list(g.groupby("langue")) + [("TOTAL", g)]:
                rows.append({"layer": layer, "speaker": speaker, "langue": language} | self._pooled(gl))
        self.per_language = pd.DataFrame(rows)
        return self.per_language

    # --- correlations -----------------------------------------------------------------

    def call_level_table(self):
        """Whole-call metrics, one row per call, columns prefixed asr_ / mt_."""
        whole = self.per_call[self.per_call.speaker == ALL]
        parts = []
        for layer, cols in CORRELATED.items():
            t = whole[whole.layer == layer].set_index("call")[cols]
            parts.append(t.add_prefix(PREFIX[layer] + "_"))
        table = pd.concat(parts, axis=1)
        table.insert(0, "langue", whole.drop_duplicates("call").set_index("call").langue)
        return table

    def correlate(self):
        table = self.call_level_table()
        columns = [c for c in table.columns if c != "langue"]
        self.correlations = spearman_matrix(table, columns)
        self.correlations_by_language = pd.concat(
            [spearman_matrix(g, columns).assign(langue=lang) for lang, g in table.groupby("langue")],
            ignore_index=True)
        return self.correlations

    def plot_correlations(self):
        """Lower-triangle Spearman matrix, Holm-corrected significance stars."""
        corr = self.correlations
        columns = list(dict.fromkeys(list(corr.metric_a) + list(corr.metric_b)))
        n = len(columns)
        rho = pd.DataFrame(np.eye(n), index=columns, columns=columns)
        p = pd.DataFrame(np.nan, index=columns, columns=columns)
        for r in corr.itertuples():
            rho.loc[r.metric_a, r.metric_b] = rho.loc[r.metric_b, r.metric_a] = r.rho
            p.loc[r.metric_a, r.metric_b] = p.loc[r.metric_b, r.metric_a] = r.p_holm
        mask = np.triu(np.ones((n, n), dtype=bool))
        # diverging: blue (negative) <-> red (positive), neutral grey midpoint
        cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32a2a"])

        fig, ax = plt.subplots(figsize=(10, 8.5))
        ax.imshow(np.ma.masked_array(rho.values, mask), cmap=cmap, vmin=-1, vmax=1)
        for i in range(n):
            for j in range(i):
                v = rho.iloc[i, j]
                ink = "white" if abs(v) > 0.6 else "#1f1f1e"
                ax.text(j, i, f"{v:.2f}\n{stars(p.iloc[i, j])}", ha="center", va="center", fontsize=6.5, color=ink)
        ax.set_xticks(range(n - 1), columns[:-1], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(1, n), columns[1:], fontsize=8)
        ax.set_xlim(-0.5, n - 1.5)
        ax.set_ylim(n - 0.5, 0.5)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
        fig.colorbar(sm, ax=ax, shrink=0.6, label="Spearman ρ")
        ax.set_title(f"Correlations between quality metrics (whole call, n = {self.corpus.call.nunique()})\n"
                     "asr_ = transcription, mt_ = translation · Holm-corrected: * p<0.05, ** p<0.01, *** p<0.001",
                     fontsize=9, color="#1f1f1e")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"correlations.{ext}", dpi=200)
        plt.close(fig)

    # --- output -----------------------------------------------------------------------

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        kw = {"index": False, "encoding": "utf-8-sig"}
        self.per_call.to_csv(OUT_DIR / "metrics_per_call.csv", **kw)
        excluded = OUT_DIR / "excluded_rows.csv"
        if len(self.excluded):
            self.excluded.to_csv(excluded, **kw)
        else:
            excluded.unlink(missing_ok=True)  # nothing excluded: no stale file from a previous run
        self.per_language.to_csv(OUT_DIR / "metrics_per_language.csv", **kw)
        self.omissions.to_csv(OUT_DIR / "omissions_per_language.csv", **kw)
        self.correlations.to_csv(OUT_DIR / "correlations.csv", **kw)
        self.correlations_by_language.to_csv(OUT_DIR / "correlations_by_language.csv", **kw)
        self.plot_correlations()
        print(f"-> {OUT_DIR}")

    def run(self):
        self.compute()
        self.summarize()
        self.deletion_runs()
        self.correlate()
        self.save()
        return self


def main():
    rq = QualityMetrics().run()
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    for layer in LAYERS:
        t = rq.per_language[rq.per_language.layer == layer].dropna(axis=1, how="all")
        print(f"\n=== {layer}\n{t.drop(columns='layer').to_string(index=False)}")
    strong = rq.correlations[rq.correlations.rho.abs() >= 0.8].sort_values("rho", key=abs, ascending=False)
    print(f"\nRedundant pairs (|rho| >= 0.8):\n{strong.to_string(index=False)}")


if __name__ == "__main__":
    main()
