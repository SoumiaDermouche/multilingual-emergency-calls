"""RQ3 - Differences across languages: do error rates and error types differ significantly across languages?

The same 20 scenarios were played in each language (17 in Spanish), so languages are compared paired by scenario:

  1. quality per language   Friedman test on the 17 scenarios played in all four languages (Kendall's W as effect
                            size), then pairwise Wilcoxon signed-rank tests (Holm-corrected within each metric);
                            Kruskal-Wallis on all 77 calls as a complement (the test of study 1).
                            Metrics of RQ1, call level, per speaker: transcription (WER, substitution, deletion and
                            insertion rates, SemDist) and translation (chrF++, SemDist).
  2. error types            chi-square test on the counts of substitutions / deletions / insertions per language,
                            per speaker (Cramer's V, adjusted standardised residuals).
  3. foreign-script turns   turns containing characters outside the Latin script (e.g. Chinese, Cyrillic), in the
                            operator's French and in the Latin-script caller languages (not Arabic): a deterministic
                            marker of hallucination, checked against the manual transcription (none expected).
                            Automatic language identification was tried and discarded: caller turns legitimately
                            contain French addresses, and short noisy turns are misidentified.
                            Test: does it explain the higher operator WER in the Arabic calls? (Fisher exact test on
                            the turns; operator WER of the Arabic calls without such turns vs the other languages.)

One caller per language: a difference between languages is also a difference between callers.

Input: resultats/rq1/metrics_per_call.csv (run rq1_metrics first).
Outputs in resultats/rq3/:
  - friedman.csv, pairwise.csv, kruskal.csv
  - error_types.csv                  counts, shares and adjusted residuals; chi2_error_types.csv
  - foreign_script_turns.csv         one row per turn (automatic and manual) with the foreign-script flag
  - foreign_script_summary.csv       count per language and speaker; foreign_script_tests.csv
  - wer_by_language.png / .pdf, error_types.png / .pdf

Run:  python -m research_questions.rq3_languages
"""

import re
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, friedmanchisquare, kruskal, mannwhitneyu, wilcoxon

from pre_annotation import LANGUAGES
from research_questions.common import RESULTS_DIR, SPEAKERS, holm, stars, study_corpus

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RQ1_FILE = RESULTS_DIR / "rq1" / "metrics_per_call.csv"
OUT_DIR = RESULTS_DIR / "rq3"

LANGUAGE_NAMES = {"anglais": "English", "arabe": "Arabic", "espagnol": "Spanish", "italien": "Italian"}
ORDER = ["arabe", "anglais", "italien", "espagnol"]
METRICS = {
    "transcription": ["wer", "sub_rate", "del_rate", "ins_rate", "semdist"],
    "translation": ["chrf", "semdist"],
}
ERROR_TYPES = {"substitutions": "Sub.", "deletions": "Del.", "insertions": "Ins."}
# a character outside Latin (basic, supplements, extended), general punctuation, digits and spaces
_FOREIGN_SCRIPT = re.compile(r"[^\x00-\u024F\u2000-\u206F\s\d]")

_TURN = re.compile(r"^(Operator|Caller):\s*(.*)$")


def turns(text):
    """(speaker, text) of each turn; unlabelled lines continue the current turn."""
    out = []
    for line in text.split("\n"):
        m = _TURN.match(line)
        if m:
            out.append([m.group(1), m.group(2)])
        elif out and line.strip():
            out[-1][1] += " " + line.strip()
    return [(s, t.replace("[non transcrit]", "").strip()) for s, t in out]


class LanguageDifferences:
    """Compare transcription and translation quality, and error types, across languages."""

    def __init__(self):
        if not RQ1_FILE.exists():
            raise FileNotFoundError(f"{RQ1_FILE} not found: run python -m research_questions.rq1_metrics first")
        self.metrics = pd.read_csv(RQ1_FILE, encoding="utf-8-sig", dtype={"scenario": str})
        self.metrics = self.metrics[self.metrics.speaker.isin(SPEAKERS)]
        self.corpus = study_corpus()

    # --- 1. quality per language ------------------------------------------------------

    def _series(self):
        """(layer, speaker, metric, wide table scenario x language)."""
        for layer, metrics in METRICS.items():
            for speaker in SPEAKERS:
                g = self.metrics[(self.metrics.layer == layer) & (self.metrics.speaker == speaker)]
                for metric in metrics:
                    yield layer, speaker, metric, g.pivot(index="scenario", columns="langue", values=metric)

    def compare(self):
        friedman, pairwise, kw = [], [], []
        for layer, speaker, metric, wide in self._series():
            key = {"layer": layer, "speaker": speaker, "metric": metric}
            complete = wide[ORDER].dropna()
            stat, p = friedmanchisquare(*[complete[l] for l in ORDER])
            n, k = len(complete), len(ORDER)
            friedman.append(key | {"n_scenarios": n, "chi2": stat, "kendall_w": stat / (n * (k - 1)), "p": p}
                            | {f"median_{l}": wide[l].median() for l in ORDER})
            rows = []
            for a, b in combinations(ORDER, 2):
                pair = wide[[a, b]].dropna()
                diff = pair[a] - pair[b]
                _, pw = wilcoxon(pair[a], pair[b]) if (diff != 0).any() else (np.nan, 1.0)
                rows.append(key | {"lang_a": a, "lang_b": b, "n": len(pair), "median_diff": diff.median(), "p": pw})
            rows = pd.DataFrame(rows)
            rows["p_holm"] = holm(rows.p)
            pairwise.append(rows)
            groups = [wide[l].dropna() for l in ORDER]
            h, pk = kruskal(*groups)
            n_all = sum(len(g) for g in groups)
            kw.append(key | {"n_calls": n_all, "H": h, "epsilon2": h / (n_all - 1), "p": pk})
        self.friedman = pd.DataFrame(friedman)
        self.friedman["p_holm"] = holm(self.friedman.p)
        self.pairwise = pd.concat(pairwise, ignore_index=True)
        self.kruskal = pd.DataFrame(kw)
        self.kruskal["p_holm"] = holm(self.kruskal.p)
        return self.friedman

    # --- 2. error types ---------------------------------------------------------------

    def error_types(self):
        rows, tests = [], []
        tr = self.metrics[self.metrics.layer == "transcription"]
        for speaker in SPEAKERS:
            counts = tr[tr.speaker == speaker].groupby("langue")[list(ERROR_TYPES)].sum().reindex(ORDER)
            chi2, p, dof, expected = chi2_contingency(counts.values)
            n = counts.values.sum()
            row_share = counts.sum(axis=1).values[:, None] / n
            col_share = counts.sum(axis=0).values[None, :] / n
            residuals = (counts.values - expected) / np.sqrt(expected * (1 - row_share) * (1 - col_share))
            cramer_v = np.sqrt(chi2 / (n * (min(counts.shape) - 1)))
            tests.append({"speaker": speaker, "chi2": chi2, "dof": dof, "p": p, "cramer_v": cramer_v, "n_errors": n})
            for i, lang in enumerate(ORDER):
                for j, error in enumerate(ERROR_TYPES):
                    rows.append({"speaker": speaker, "langue": lang, "error": error,
                                 "count": counts.iloc[i, j], "share": counts.iloc[i, j] / counts.iloc[i].sum(),
                                 "adjusted_residual": residuals[i, j]})
        self.error_counts = pd.DataFrame(rows)
        self.chi2 = pd.DataFrame(tests)
        return self.chi2

    # --- 3. foreign-script turns -----------------------------------------------------

    def foreign_script(self):
        rows = []
        for r in self.corpus.itertuples():
            for version, text in (("auto", r.transcription_auto), ("manual", r.transcription_manuelle)):
                for i, (speaker, turn) in enumerate(turns(text)):
                    if speaker == "Caller" and r.langue == "arabe":
                        continue  # Arabic script is expected, Latin script (French addresses) too
                    rows.append({"call": r.call, "langue": r.langue, "version": version, "turn": i + 1,
                                 "speaker": speaker, "foreign_script": bool(_FOREIGN_SCRIPT.search(turn)),
                                 "text": turn})
        self.script_turns = pd.DataFrame(rows)
        summary = self.script_turns.groupby(["version", "speaker", "langue"]).agg(
            turns=("foreign_script", "size"), foreign_script=("foreign_script", "sum"),
            calls=("call", lambda c: c[self.script_turns.loc[c.index, "foreign_script"]].nunique())).reset_index()
        self.script_summary = summary

        # tests on the operator side, automatic transcription
        op = summary[(summary.version == "auto") & (summary.speaker == "Operator")].set_index("langue")
        arabic = op.loc["arabe"]
        others = op.drop(index="arabe").sum()
        table = [[arabic.foreign_script, arabic.turns - arabic.foreign_script],
                 [others.foreign_script, others.turns - others.foreign_script]]
        odds, p_fisher = fisher_exact(table)
        tests = [{"test": "Fisher: operator foreign-script turns, Arabic calls vs other languages",
                  "statistic": odds, "p": p_fisher}]
        flagged = set(self.script_turns[(self.script_turns.version == "auto") & self.script_turns.foreign_script
                                        & (self.script_turns.speaker == "Operator")].call)
        wer = self.metrics[(self.metrics.layer == "transcription") & (self.metrics.speaker == "Operator")]
        clean_arabic = wer[(wer.langue == "arabe") & ~wer.call.isin(flagged)].wer
        other = wer[wer.langue != "arabe"].wer
        u, p_mw = mannwhitneyu(clean_arabic, other, alternative="two-sided")
        tests.append({"test": f"Mann-Whitney: operator WER, Arabic calls without foreign-script turns "
                              f"(n = {len(clean_arabic)}, median {clean_arabic.median():.3f}) vs other languages "
                              f"(n = {len(other)}, median {other.median():.3f})", "statistic": u, "p": p_mw})
        self.script_tests = pd.DataFrame(tests)
        return summary

    # --- figures ----------------------------------------------------------------------

    def plot_wer(self):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        tr = self.metrics[self.metrics.layer == "transcription"]
        for ax, speaker in zip(axes, SPEAKERS):
            g = tr[tr.speaker == speaker]
            ax.boxplot([g[g.langue == l].wer * 100 for l in ORDER], showfliers=True)
            ax.set_xticks(range(1, 5), [LANGUAGE_NAMES[l] for l in ORDER])
            f = self.friedman[(self.friedman.layer == "transcription") & (self.friedman.speaker == speaker)
                              & (self.friedman.metric == "wer")].iloc[0]
            ax.set_title(f"{speaker}: Friedman p = {f.p:.2g} (Holm: {f.p_holm:.2g}{stars(f.p_holm)}), "
                         f"W = {f.kendall_w:.2f}", fontsize=9)
        axes[0].set_ylabel("WER per call (%)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"wer_by_language.{ext}", dpi=200)
        plt.close(fig)

    def plot_error_types(self):
        colors = {"substitutions": "#4c72b0", "deletions": "#dd8452", "insertions": "#55a868"}
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, speaker in zip(axes, SPEAKERS):
            g = self.error_counts[self.error_counts.speaker == speaker]
            bottom = np.zeros(len(ORDER))
            for error in ERROR_TYPES:
                share = g[g.error == error].set_index("langue").reindex(ORDER).share.values * 100
                ax.bar(range(len(ORDER)), share, bottom=bottom, color=colors[error], label=error)
                for x, (b, s) in enumerate(zip(bottom, share)):
                    if s >= 6:
                        ax.text(x, b + s / 2, f"{s:.0f}", ha="center", va="center", fontsize=8, color="white")
                bottom += share
            c = self.chi2[self.chi2.speaker == speaker].iloc[0]
            ax.set_title(f"{speaker}: chi2 p = {c.p:.2g}, V = {c.cramer_v:.2f}", fontsize=9)
            ax.set_xticks(range(len(ORDER)), [LANGUAGE_NAMES[l] for l in ORDER])
        axes[0].set_ylabel("Share of transcription errors (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"error_types.{ext}", dpi=200)
        plt.close(fig)

    # --- output -----------------------------------------------------------------------

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        kw = {"index": False, "encoding": "utf-8-sig"}
        self.friedman.to_csv(OUT_DIR / "friedman.csv", **kw)
        self.pairwise.to_csv(OUT_DIR / "pairwise.csv", **kw)
        self.kruskal.to_csv(OUT_DIR / "kruskal.csv", **kw)
        self.error_counts.to_csv(OUT_DIR / "error_types.csv", **kw)
        self.chi2.to_csv(OUT_DIR / "chi2_error_types.csv", **kw)
        self.script_turns.to_csv(OUT_DIR / "foreign_script_turns.csv", **kw)
        self.script_summary.to_csv(OUT_DIR / "foreign_script_summary.csv", **kw)
        self.script_tests.to_csv(OUT_DIR / "foreign_script_tests.csv", **kw)
        self.plot_wer()
        self.plot_error_types()
        print(f"-> {OUT_DIR}")

    def run(self):
        self.compare()
        self.error_types()
        self.foreign_script()
        self.save()
        return self


def main():
    rq = LanguageDifferences().run()
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 30,
                  "display.float_format", "{:.3f}".format)
    print(f"\n=== Friedman (17 scenarios)\n{rq.friedman.to_string(index=False)}")
    sig = rq.pairwise[rq.pairwise.p_holm < 0.05]
    print(f"\n=== significant pairwise Wilcoxon (Holm within metric)\n{sig.to_string(index=False)}")
    print(f"\n=== Kruskal-Wallis (77 calls)\n{rq.kruskal.to_string(index=False)}")
    print(f"\n=== error types\n{rq.chi2.to_string(index=False)}")
    print(rq.error_counts.pivot_table(index=['speaker', 'langue'], columns='error',
                                      values=['share', 'adjusted_residual']).round(2).to_string())
    print(f"\n=== foreign-script turns\n{rq.script_summary.to_string(index=False)}")
    print(rq.script_tests.to_string(index=False))


if __name__ == "__main__":
    main()
