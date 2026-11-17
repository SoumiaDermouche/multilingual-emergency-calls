"""Trends in the corpus - descriptive figures (no statistical test).

  1. Pareto per call       calls sorted by number of errors, cumulative share of the errors; share of the calls
                           that concentrate 80% of the errors, Gini coefficient. Transcription: substitutions +
                           deletions + insertions (normalised words, as in RQ1). Translation: TER edits against the
                           manual translation (raw text, as in RQ1). Per speaker; bars coloured by language.
  2. word clouds           words of the reference most affected (hyphenated names kept whole): transcription =
                           reference words substituted or omitted by Whisper (jiwer alignment); translation =
                           words of the manual translation missing from the automatic one (multiset difference per
                           call). Stop words and fillers removed.
                           Operator (French) and caller (per language); translation into French and into each
                           caller language.
  3. language x emergency  heatmap of the pooled WER (operator, caller) per language and emergency type.

Outputs in resultats/trends/:
  - pareto_calls.csv, pareto_summary.csv, pareto.png / .pdf
  - affected_words.csv (word counts), wordclouds_transcription.png / .pdf, wordclouds_translation.png / .pdf
  - heatmap_language_type.csv, heatmap_language_type.png / .pdf

Run:  python -m research_questions.trends
"""

import re
from collections import Counter

import jiwer
import matplotlib
import numpy as np
import pandas as pd
from sacrebleu.metrics import TER

from pre_annotation import LANGUAGES, normalize
from research_questions.common import RESULTS_DIR, SPEAKERS, split_speakers, study_corpus

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = RESULTS_DIR / "trends"
LANGUAGE_NAMES = {"anglais": "English", "arabe": "Arabic", "espagnol": "Spanish", "italien": "Italian"}
ORDER = ["arabe", "anglais", "italien", "espagnol"]
COLORS = {"arabe": "#1f77b4", "anglais": "#ff7f0e", "italien": "#2ca02c", "espagnol": "#d62728"}
TYPES = ["incendie", "secours à la personne", "accident"]
TYPE_NAMES = {"incendie": "Fire", "secours à la personne": "Medical", "accident": "Road accident"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # has Arabic glyphs
# fillers and interjections, not in the stop-word lists
FILLERS = {"euh", "heu", "hum", "ah", "oh", "eh", "ok", "okay", "uh", "um", "mm", "hmm", "mmm", "ehm", "allo", "allô"}
TOP_WORDS = 60
# for the word clouds, hyphenated names stay one word ("Bourg-en-Bresse", not "bourg", "en", "bresse"): normalize()
# drops non-alphanumeric characters, so the hyphen is protected by a letter that never occurs in the corpus
_HYPHEN = "ʬ"
_HYPHENATED = re.compile(r"(?<=\w)-(?=\w)")


def cloud_normalize(text):
    return normalize(_HYPHENATED.sub(_HYPHEN, text))


def stop_words(code):
    from stop_words import get_stop_words

    words = set()
    for w in get_stop_words(code):
        words |= set(normalize(w).split())
    return words | FILLERS


class CorpusTrends:
    """Descriptive trends: concentration of the errors, most affected words, language x emergency type."""

    def __init__(self):
        self.corpus = study_corpus()
        self.texts = []
        for r in self.corpus.itertuples():
            tm, ta = split_speakers(r.transcription_manuelle), split_speakers(r.transcription_auto)
            mm, ma = split_speakers(r.traduction_manuelle), split_speakers(r.traduction_auto)
            for speaker in SPEAKERS:
                self.texts.append({"call": r.call, "langue": r.langue, "type": r.categorie_urgence,
                                   "speaker": speaker, "tr_ref": tm[speaker], "tr_hyp": ta[speaker],
                                   "mt_ref": mm[speaker], "mt_hyp": ma[speaker]})
        self.texts = pd.DataFrame(self.texts)

    # --- 1. Pareto --------------------------------------------------------------------

    def errors_per_call(self):
        ter = TER()
        rows = []
        for t in self.texts.itertuples():
            row = {"call": t.call, "langue": t.langue, "speaker": t.speaker}
            if normalize(t.tr_ref):
                o = jiwer.process_words(normalize(t.tr_ref), normalize(t.tr_hyp) or "∅")
                row |= {"tr_errors": o.substitutions + o.deletions + o.insertions,
                        "tr_ref_words": o.hits + o.substitutions + o.deletions}
            if t.mt_ref.strip():
                s = ter.sentence_score(t.mt_hyp, [t.mt_ref])
                row |= {"mt_errors": s.num_edits, "mt_ref_words": s.ref_length}
            rows.append(row)
        self.errors = pd.DataFrame(rows)
        return self.errors

    @staticmethod
    def _concentration(errors):
        e = np.sort(errors.dropna().values)[::-1]
        cum = np.cumsum(e) / e.sum()
        n = len(e)
        calls_80 = int(np.searchsorted(cum, 0.8) + 1)
        top20 = cum[int(np.ceil(0.2 * n)) - 1]
        asc = e[::-1]
        gini = (2 * np.sum(np.arange(1, n + 1) * asc) / (n * asc.sum())) - (n + 1) / n
        return {"calls": n, "errors": e.sum(), "calls_for_80pct": calls_80, "share_calls_for_80pct": calls_80 / n,
                "share_errors_top20pct_calls": top20, "gini": gini}

    def pareto(self):
        rows = []
        for layer in ("tr", "mt"):
            for speaker in SPEAKERS:
                g = self.errors[self.errors.speaker == speaker]
                rows.append({"layer": "transcription" if layer == "tr" else "translation", "speaker": speaker}
                            | self._concentration(g[f"{layer}_errors"]))
        self.pareto_summary = pd.DataFrame(rows)
        return self.pareto_summary

    def plot_pareto(self):
        panels = [("tr", "Operator", "Transcription, operator (French)"),
                  ("tr", "Caller", "Transcription, caller"),
                  ("mt", "Caller", "Translation, caller -> French"),
                  ("mt", "Operator", "Translation, operator -> caller language")]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        for ax, (layer, speaker, title) in zip(axes.flat, panels):
            g = self.errors[self.errors.speaker == speaker].dropna(subset=[f"{layer}_errors"])
            g = g.sort_values(f"{layer}_errors", ascending=False)
            e = g[f"{layer}_errors"].values
            x = np.arange(1, len(e) + 1)
            ax.bar(x, e, color=[COLORS[l] for l in g.langue], width=0.9)
            ax.set_xlim(0, len(e) + 1)
            ax.set_ylabel("Errors per call")
            ax2 = ax.twinx()
            cum = np.cumsum(e) / e.sum() * 100
            ax2.plot(x, cum, color="black", lw=1.3)
            ax2.axhline(80, color="grey", ls="--", lw=0.8)
            name = "transcription" if layer == "tr" else "translation"
            ps = self.pareto_summary
            s = ps[(ps.layer == name) & (ps.speaker == speaker)].iloc[0]
            ax2.axvline(s.calls_for_80pct, color="grey", ls="--", lw=0.8)
            ax2.set_ylim(0, 105)
            ax2.set_ylabel("Cumulative share (%)")
            ax.set_title(f"{title}\n80% of the errors in {s.calls_for_80pct} of {s.calls} calls "
                         f"({s.share_calls_for_80pct:.0%}); Gini = {s.gini:.2f}", fontsize=9)
            ax.set_xlabel("Calls, sorted by number of errors")
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in ORDER]
        fig.legend(handles, [LANGUAGE_NAMES[l] for l in ORDER], loc="lower center", ncol=4, fontsize=9,
                   frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"pareto.{ext}", dpi=200)
        plt.close(fig)

    # --- 2. word clouds ---------------------------------------------------------------

    @staticmethod
    def _asr_affected(ref, hyp):
        ref_n, hyp_n = cloud_normalize(ref), cloud_normalize(hyp) or "∅"
        if not ref_n:
            return []
        words = ref_n.split()
        out = []
        for chunk in jiwer.process_words(ref_n, hyp_n).alignments[0]:
            if chunk.type in ("substitute", "delete"):
                out += words[chunk.ref_start_idx:chunk.ref_end_idx]
        return out

    @staticmethod
    def _mt_missing(ref, hyp):
        missing = Counter(cloud_normalize(ref).split()) - Counter(cloud_normalize(hyp).split())
        return list(missing.elements())

    def affected_words(self):
        rows = []
        for t in self.texts.itertuples():
            spoken = "fr" if t.speaker == "Operator" else LANGUAGES[t.langue]
            translated = LANGUAGES[t.langue] if t.speaker == "Operator" else "fr"
            for layer, words, code in (("transcription", self._asr_affected(t.tr_ref, t.tr_hyp), spoken),
                                       ("translation", self._mt_missing(t.mt_ref, t.mt_hyp), translated)):
                for w in words:
                    rows.append({"layer": layer, "speaker": t.speaker, "langue": t.langue, "text_language": code,
                                 "word": w})
        words = pd.DataFrame(rows)
        words["word"] = words.word.str.replace(_HYPHEN, "-")
        sw = {code: stop_words(code) for code in set(words.text_language)}
        # a hyphenated form made only of stop words ("a-t-il", "est-ce") is a stop word too
        keep = [not all(p in sw[c] or len(p) == 1 for p in w.split("-")) and (len(w) > 1 or w.isdigit())
                for w, c in zip(words.word, words.text_language)]
        words = words[keep]
        self.words = (words.groupby(["layer", "text_language", "speaker", "langue", "word"]).size()
                      .rename("count").reset_index().sort_values("count", ascending=False))
        return self.words

    @staticmethod
    def _display(word, code):
        """Arabic needs letter shaping and right-to-left order: done by Pillow itself when it has libraqm,
        otherwise by arabic_reshaper + python-bidi (doing both would break the words)."""
        from PIL import features

        if code != "ar" or features.check("raqm"):
            return word
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(word))

    def _cloud(self, ax, counts, code, title):
        from wordcloud import WordCloud

        freqs = {self._display(w, code): c for w, c in counts.head(TOP_WORDS).items()}
        if freqs:
            wc = WordCloud(width=800, height=450, background_color="white", font_path=FONT,
                           colormap="viridis", prefer_horizontal=0.95, random_state=0)
            ax.imshow(wc.generate_from_frequencies(freqs), interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def plot_clouds(self):
        for layer in ("transcription", "translation"):
            w = self.words[self.words.layer == layer]
            panels = []
            if layer == "transcription":
                op = w[w.speaker == "Operator"].groupby("word")["count"].sum().sort_values(ascending=False)
                panels.append((op, "fr", "Operator (French), all calls"))
                for l in ORDER:
                    c = w[(w.speaker == "Caller") & (w.langue == l)].set_index("word")["count"]
                    panels.append((c.sort_values(ascending=False), LANGUAGES[l], f"Caller, {LANGUAGE_NAMES[l]}"))
                suptitle = "Words of the corrected transcription substituted or omitted by Whisper"
            else:
                fr = w[w.speaker == "Caller"].groupby("word")["count"].sum().sort_values(ascending=False)
                panels.append((fr, "fr", "Caller -> French, all calls"))
                for l in ORDER:
                    c = w[(w.speaker == "Operator") & (w.langue == l)].set_index("word")["count"]
                    panels.append((c.sort_values(ascending=False), LANGUAGES[l],
                                   f"Operator -> {LANGUAGE_NAMES[l]}"))
                suptitle = "Words of the manual translation missing from the automatic translation"
            fig, axes = plt.subplots(2, 3, figsize=(13, 5.6))
            axes = axes.flat
            self._cloud(axes[0], *panels[0])
            axes[3].axis("off")
            for ax, panel in zip([axes[1], axes[2], axes[4], axes[5]], panels[1:]):
                self._cloud(ax, *panel)
            fig.suptitle(f"{suptitle} (size = number of occurrences; stop words removed)", fontsize=10)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(OUT_DIR / f"wordclouds_{layer}.{ext}", dpi=200)
            plt.close(fig)

    # --- 3. language x emergency type -------------------------------------------------

    def heatmap(self):
        e = self.errors.merge(self.corpus[["call", "categorie_urgence"]], on="call")
        rows = []
        for speaker in SPEAKERS:
            for l in ORDER:
                for k in TYPES:
                    g = e[(e.speaker == speaker) & (e.langue == l) & (e.categorie_urgence == k)]
                    rows.append({"speaker": speaker, "langue": l, "type": k, "calls": len(g),
                                 "wer": g.tr_errors.sum() / g.tr_ref_words.sum()})
        self.cells = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, speaker in zip(axes, SPEAKERS):
            c = self.cells[self.cells.speaker == speaker]
            grid = c.pivot(index="langue", columns="type", values="wer").reindex(index=ORDER, columns=TYPES) * 100
            n = c.pivot(index="langue", columns="type", values="calls").reindex(index=ORDER, columns=TYPES)
            im = ax.imshow(grid.values, cmap="Reds", vmin=0, vmax=50)
            for i in range(len(ORDER)):
                for j in range(len(TYPES)):
                    v = grid.values[i, j]
                    ax.text(j, i, f"{v:.1f}\n(n = {int(n.values[i, j])})", ha="center", va="center", fontsize=8,
                            color="white" if v > 30 else "black")
            ax.set_xticks(range(len(TYPES)), [TYPE_NAMES[k] for k in TYPES], fontsize=8)
            ax.set_yticks(range(len(ORDER)), [LANGUAGE_NAMES[l] for l in ORDER], fontsize=8)
            ax.set_title(f"WER, {speaker.lower()} (%)", fontsize=9)
        fig.colorbar(im, ax=axes, shrink=0.8, label="WER (%)")
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"heatmap_language_type.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return self.cells

    # --- output -----------------------------------------------------------------------

    def run(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        kw = {"index": False, "encoding": "utf-8-sig"}
        self.errors_per_call()
        self.pareto()
        self.affected_words()
        self.errors.to_csv(OUT_DIR / "pareto_calls.csv", **kw)
        self.pareto_summary.to_csv(OUT_DIR / "pareto_summary.csv", **kw)
        self.words.to_csv(OUT_DIR / "affected_words.csv", **kw)
        self.plot_pareto()
        self.plot_clouds()
        self.heatmap().to_csv(OUT_DIR / "heatmap_language_type.csv", **kw)
        print(f"-> {OUT_DIR}")
        return self


def main():
    t = CorpusTrends().run()
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(f"\n=== Pareto\n{t.pareto_summary.to_string(index=False)}")
    for (layer, speaker), g in t.words.groupby(["layer", "speaker"]):
        top = g.groupby("word")["count"].sum().sort_values(ascending=False).head(15)
        print(f"\n=== {layer} / {speaker}: {', '.join(f'{w} ({c})' for w, c in top.items())}")
    print(f"\n=== WER language x type\n{t.cells.pivot_table(index=['speaker', 'langue'], columns='type', values='wer')}")


if __name__ == "__main__":
    main()
