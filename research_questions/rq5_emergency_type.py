"""RQ5 - Emergency type: does the type of emergency (fire, medical assistance, road accident) influence the measured
quality and its evaluation?

The calls of each emergency type are pooled across languages and the three groups are compared (as in study 1):

  1. measured quality   Kruskal-Wallis test per call-level metric of RQ1 (transcription: WER, substitution,
                        deletion and insertion rates, SemDist; translation: chrF++, SemDist; per speaker), epsilon^2
                        as effect size, Holm-corrected; pairwise Mann-Whitney tests (Holm within metric).
                        Robustness check: the same test on within-language ranks (removes the language effect,
                        e.g. the Arabic calls for the operator).
  2. error types        chi-square test on the counts of substitutions / deletions / insertions per type.
  3. evaluation         Kruskal-Wallis on the questionnaire scores (replication of study 1 on 77 calls);
                        agreement between the transcription score and the measured transcription quality in each
                        type (within-language Spearman, as in RQ2); operator - caller gap per type (Wilcoxon) and
                        across types (Kruskal-Wallis).

Input: resultats/rq1/metrics_per_call.csv (run rq1_metrics first).
Outputs in resultats/rq5/:
  - quality.csv, quality_pairwise.csv, error_types.csv, chi2_error_types.csv
  - scores.csv, agreement_by_type.csv, gap_by_type.csv
  - quality_by_type.png / .pdf, error_types_by_type.png / .pdf

Run:  python -m research_questions.rq5_emergency_type
"""

from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr, wilcoxon

from corpus import DIMENSIONS, DIMENSIONS_BY_ROLE
from research_questions.common import RESULTS_DIR, SPEAKERS, holm, stars
from research_questions.rq2_perceived_vs_measured import PerceivedVsMeasured

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = RESULTS_DIR / "rq5"
TYPES = ["incendie", "secours à la personne", "accident"]
TYPE_NAMES = {"incendie": "Fire", "secours à la personne": "Medical", "accident": "Road accident"}
METRICS = {"transcription": ["wer", "sub_rate", "del_rate", "ins_rate", "semdist"], "translation": ["chrf", "semdist"]}
ERROR_TYPES = ["substitutions", "deletions", "insertions"]
SHARED = ["transcription", "traduction_vers_fr", "compris", "confiance"]
ROLE_NAMES = {"appelant": "caller", "operateur": "operator"}


class EmergencyType:
    """Compare measured quality, error types and evaluation across emergency types."""

    def __init__(self):
        base = PerceivedVsMeasured()
        self.metrics = base.metrics[base.metrics.speaker.isin(SPEAKERS)]
        self.table = base.table.join(base.corpus.set_index("call").categorie_urgence)

    @staticmethod
    def _kruskal(values, groups):
        samples = [values[groups == t].dropna() for t in TYPES]
        h, p = kruskal(*samples)
        n = sum(len(s) for s in samples)
        return h, p, h / (n - 1), n

    # --- 1. measured quality ----------------------------------------------------------

    def quality(self):
        rows, pairs = [], []
        for layer, metrics in METRICS.items():
            for speaker in SPEAKERS:
                g = self.metrics[(self.metrics.layer == layer) & (self.metrics.speaker == speaker)]
                g = g.set_index("call")
                for metric in metrics:
                    v, groups = g[metric], g.categorie_urgence
                    h, p, eps, n = self._kruskal(v, groups)
                    within = g.groupby("langue")[metric].rank(pct=True)
                    h_w, p_w, _, _ = self._kruskal(within, groups)
                    rows.append({"layer": layer, "speaker": speaker, "metric": metric, "n": n, "H": h,
                                 "epsilon2": eps, "p": p, "p_within_language": p_w}
                                | {f"median_{t}": v[groups == t].median() for t in TYPES})
                    pw = []
                    for a, b in combinations(TYPES, 2):
                        _, pm = mannwhitneyu(v[groups == a].dropna(), v[groups == b].dropna())
                        pw.append({"layer": layer, "speaker": speaker, "metric": metric, "type_a": a,
                                   "type_b": b, "p": pm})
                    pw = pd.DataFrame(pw)
                    pw["p_holm"] = holm(pw.p)
                    pairs.append(pw)
        self.quality_tests = pd.DataFrame(rows)
        self.quality_tests["p_holm"] = holm(self.quality_tests.p)
        self.quality_tests["p_within_language_holm"] = holm(self.quality_tests.p_within_language)
        self.quality_pairs = pd.concat(pairs, ignore_index=True)
        return self.quality_tests

    # --- 2. error types ---------------------------------------------------------------

    def error_types(self):
        rows, tests = [], []
        tr = self.metrics[self.metrics.layer == "transcription"]
        for speaker in SPEAKERS:
            counts = tr[tr.speaker == speaker].groupby("categorie_urgence")[ERROR_TYPES].sum().reindex(TYPES)
            chi2, p, dof, expected = chi2_contingency(counts.values)
            n = counts.values.sum()
            r = counts.sum(axis=1).values[:, None] / n
            c = counts.sum(axis=0).values[None, :] / n
            residuals = (counts.values - expected) / np.sqrt(expected * (1 - r) * (1 - c))
            tests.append({"speaker": speaker, "chi2": chi2, "dof": dof, "p": p,
                          "cramer_v": np.sqrt(chi2 / (n * (min(counts.shape) - 1))), "n_errors": n})
            for i, t in enumerate(TYPES):
                for j, e in enumerate(ERROR_TYPES):
                    rows.append({"speaker": speaker, "type": t, "error": e, "count": counts.iloc[i, j],
                                 "share": counts.iloc[i, j] / counts.iloc[i].sum(),
                                 "adjusted_residual": residuals[i, j]})
        self.error_counts = pd.DataFrame(rows)
        self.chi2 = pd.DataFrame(tests)
        return self.chi2

    # --- 3. evaluation ----------------------------------------------------------------

    def evaluation(self):
        t, groups = self.table, self.table.categorie_urgence
        rows = []
        for role, dims in DIMENSIONS_BY_ROLE.items():
            for dim in dims:
                col = f"{role}_{dim}"
                h, p, eps, n = self._kruskal(t[col], groups)
                rows.append({"role": ROLE_NAMES[role], "dimension": DIMENSIONS[dim], "n": n, "H": h,
                             "epsilon2": eps, "p": p} | {f"mean_{k}": t.loc[groups == k, col].mean() for k in TYPES})
        self.scores = pd.DataFrame(rows)
        self.scores["p_holm"] = holm(self.scores.p)

        rows = []
        for k in TYPES:
            g = t[groups == k]
            for role in DIMENSIONS_BY_ROLE:
                for measure in ("asr_wer_call", "asr_semdist_call"):
                    cols = [f"{role}_transcription", measure]
                    r = g.groupby("langue")[cols].rank(pct=True)
                    rho, p = spearmanr(r[cols[0]], r[cols[1]])
                    rows.append({"type": k, "role": ROLE_NAMES[role], "score": "transcription_quality",
                                 "measure": measure, "n": len(g), "agreement": -rho, "p": p})
        self.agreement = pd.DataFrame(rows)
        self.agreement["p_holm"] = holm(self.agreement.p)

        rows = []
        for dim in SHARED:
            gap = t[f"operateur_{dim}"] - t[f"appelant_{dim}"]
            h, p, eps, n = self._kruskal(gap, groups)
            rows.append({"type": "across types (Kruskal-Wallis)", "dimension": DIMENSIONS[dim], "n": n,
                         "mean_gap": gap.mean(), "statistic": h, "p": p})
            for k in TYPES:
                op, ca = t.loc[groups == k, f"operateur_{dim}"], t.loc[groups == k, f"appelant_{dim}"]
                stat, pw = wilcoxon(op, ca) if (op != ca).any() else (np.nan, 1.0)
                rows.append({"type": k, "dimension": DIMENSIONS[dim], "n": len(op), "mean_gap": (op - ca).mean(),
                             "statistic": stat, "p": pw})
        self.gaps = pd.DataFrame(rows)
        across = self.gaps.type.str.startswith("across")
        self.gaps.loc[across, "p_holm"] = holm(self.gaps.loc[across, "p"])
        self.gaps.loc[~across, "p_holm"] = holm(self.gaps.loc[~across, "p"])
        return self.scores

    # --- figures ----------------------------------------------------------------------

    def plot_quality(self):
        panels = [("transcription", "Operator", "wer", "Operator WER (%)", 100),
                  ("transcription", "Caller", "wer", "Caller WER (%)", 100),
                  ("translation", "Caller", "semdist", "SemDist, caller -> French", 1)]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, (layer, speaker, metric, label, scale) in zip(axes, panels):
            g = self.metrics[(self.metrics.layer == layer) & (self.metrics.speaker == speaker)]
            ax.boxplot([g[g.categorie_urgence == k][metric] * scale for k in TYPES])
            n = [int((g.categorie_urgence == k).sum()) for k in TYPES]
            ax.set_xticks(range(1, 4), [f"{TYPE_NAMES[k]}\n(n = {m})" for k, m in zip(TYPES, n)], fontsize=8)
            q = self.quality_tests[(self.quality_tests.layer == layer) & (self.quality_tests.speaker == speaker)
                                   & (self.quality_tests.metric == metric)].iloc[0]
            ax.set_title(f"{label}\nKruskal-Wallis p = {q.p:.2g} (Holm: {q.p_holm:.2g}{stars(q.p_holm)})",
                         fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"quality_by_type.{ext}", dpi=200)
        plt.close(fig)

    def plot_error_types(self):
        colors = {"substitutions": "#4c72b0", "deletions": "#dd8452", "insertions": "#55a868"}
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, speaker in zip(axes, SPEAKERS):
            g = self.error_counts[self.error_counts.speaker == speaker]
            bottom = np.zeros(len(TYPES))
            for e in ERROR_TYPES:
                share = g[g.error == e].set_index("type").reindex(TYPES).share.values * 100
                ax.bar(range(len(TYPES)), share, bottom=bottom, color=colors[e], label=e)
                for x, (b, s) in enumerate(zip(bottom, share)):
                    if s >= 6:
                        ax.text(x, b + s / 2, f"{s:.0f}", ha="center", va="center", fontsize=8, color="white")
                bottom += share
            c = self.chi2[self.chi2.speaker == speaker].iloc[0]
            ax.set_title(f"{speaker}: chi2 p = {c.p:.2g}, V = {c.cramer_v:.2f}", fontsize=9)
            ax.set_xticks(range(len(TYPES)), [TYPE_NAMES[k] for k in TYPES])
        axes[0].set_ylabel("Share of transcription errors (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"error_types_by_type.{ext}", dpi=200)
        plt.close(fig)

    # --- output -----------------------------------------------------------------------

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        kw = {"index": False, "encoding": "utf-8-sig"}
        self.quality_tests.to_csv(OUT_DIR / "quality.csv", **kw)
        self.quality_pairs.to_csv(OUT_DIR / "quality_pairwise.csv", **kw)
        self.error_counts.to_csv(OUT_DIR / "error_types.csv", **kw)
        self.chi2.to_csv(OUT_DIR / "chi2_error_types.csv", **kw)
        self.scores.to_csv(OUT_DIR / "scores.csv", **kw)
        self.agreement.to_csv(OUT_DIR / "agreement_by_type.csv", **kw)
        self.gaps.to_csv(OUT_DIR / "gap_by_type.csv", **kw)
        self.plot_quality()
        self.plot_error_types()
        print(f"-> {OUT_DIR}")

    def run(self):
        self.quality()
        self.error_types()
        self.evaluation()
        self.save()
        return self


def main():
    rq = EmergencyType().run()
    pd.set_option("display.width", 250, "display.max_rows", 100, "display.max_columns", 30,
                  "display.float_format", "{:.3f}".format)
    print(f"\n=== measured quality\n{rq.quality_tests.to_string(index=False)}")
    sig = rq.quality_pairs[rq.quality_pairs.p_holm < 0.05]
    print(f"\n=== significant pairwise (Holm within metric)\n{sig.to_string(index=False)}")
    print(f"\n=== error types\n{rq.chi2.to_string(index=False)}")
    print(rq.error_counts.pivot_table(index=["speaker", "type"], columns="error",
                                      values=["share", "adjusted_residual"]).round(2).to_string())
    print(f"\n=== scores\n{rq.scores.to_string(index=False)}")
    print(f"\n=== agreement by type\n{rq.agreement.to_string(index=False)}")
    print(f"\n=== gap by type\n{rq.gaps.to_string(index=False)}")


if __name__ == "__main__":
    main()
