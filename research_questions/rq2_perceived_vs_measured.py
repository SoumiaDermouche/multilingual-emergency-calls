"""RQ2 - Perceived vs. measured quality: do the questionnaire scores reflect the errors actually measured?

Each questionnaire score (study 1, 1-5 scale) is compared with the reference-based metrics of RQ1 that measure the
same thing:

  transcription score                 transcription metrics of the whole call, of the operator and of the caller turns
  translation into French score       translation metrics of the caller turns (caller -> French)
  translation into caller language    translation metrics of the operator turns (French -> caller language), caller only
  feeling understood, confidence      transcription and translation metrics of the whole call

Correlations are Spearman's rho at call level, Holm-corrected within each family:
  - pooled: all 77 calls;
  - within language: scores and metrics are ranked within each language before correlating, which removes the
    differences between languages. There is one caller per language, so the pooled correlations mix the effect of
    the errors with the differences between callers (and languages); the within-language ones do not.

Input: resultats/rq1/metrics_per_call.csv (run rq1_metrics first).
Outputs in resultats/rq2/:
  - correlations.csv                 score x metric, pooled and within language
  - correlations_by_language.csv     same, computed in each language (n = 17-20, exploratory)
  - per_language.csv                 mean score and corpus-level metric per language and role
  - omissions_and_translation.csv    why translation scores and metrics disagree: effect of the transcription
                                     omissions on the measured translation quality, and score / metric agreement
                                     once the omissions are controlled for (within language)
  - score_distribution.csv           number of calls per score level (ceiling effect)
  - correlations.png / .pdf          heatmap of the within-language correlations
  - scores_vs_semdist.png / .pdf     distribution of the measured error by questionnaire score

Run:  python -m research_questions.rq2_perceived_vs_measured
"""

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from corpus import DIMENSIONS, DIMENSIONS_BY_ROLE
from research_questions.common import ALL, RESULTS_DIR, holm, stars, study_corpus

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RQ1_FILE = RESULTS_DIR / "rq1" / "metrics_per_call.csv"
OUT_DIR = RESULTS_DIR / "rq2"

ROLE_NAMES = {"appelant": "caller", "operateur": "operator"}
ASR_METRICS = ["wer", "sub_rate", "del_rate", "semdist", "bertscore_f1"]
MT_METRICS = ["chrf", "semdist", "bertscore_f1"]
# metrics where a higher value means a better output (expected positive correlation with the score)
HIGHER_IS_BETTER = {"bertscore_f1", "chrf", "bleu"}

# score dimension -> measured texts it refers to: (layer, speaker) of the RQ1 metrics
TARGETS = {
    "transcription": [("transcription", ALL), ("transcription", "Operator"), ("transcription", "Caller")],
    "traduction_vers_fr": [("translation", "Caller")],
    "traduction_vers_appelant": [("translation", "Operator")],
    "compris": [("transcription", ALL), ("translation", ALL)],
    "confiance": [("transcription", ALL), ("translation", ALL)],
}
LAYER_PREFIX = {"transcription": "asr", "translation": "mt"}
SPEAKER_SUFFIX = {ALL: "call", "Operator": "operator", "Caller": "caller"}


def measure_name(layer, speaker, metric):
    return f"{LAYER_PREFIX[layer]}_{metric}_{SPEAKER_SUFFIX[speaker]}"


class PerceivedVsMeasured:
    """Correlate the questionnaire scores with the reference-based metrics of RQ1."""

    def __init__(self):
        if not RQ1_FILE.exists():
            raise FileNotFoundError(f"{RQ1_FILE} not found: run python -m research_questions.rq1_metrics first")
        self.metrics = pd.read_csv(RQ1_FILE, encoding="utf-8-sig")
        self.corpus = study_corpus()
        self.table = self._call_table()
        self.correlations = None
        self.correlations_by_language = None
        self.per_language = None

    # --- data -------------------------------------------------------------------------

    def _call_table(self):
        """One row per call: language, questionnaire scores and the measured metrics (wide format)."""
        table = self.corpus.set_index("call")[["langue"] + [f"{r}_{d}" for r in DIMENSIONS_BY_ROLE
                                                            for d in DIMENSIONS_BY_ROLE[r]]].copy()
        for col in table.columns[1:]:
            table[col] = table[col].astype(float)
        for (layer, speaker), g in self.metrics.groupby(["layer", "speaker"]):
            cols = ASR_METRICS if layer == "transcription" else MT_METRICS
            g = g.set_index("call")[cols]
            table = table.join(g.rename(columns={m: measure_name(layer, speaker, m) for m in cols}))
        return table

    def pairs(self):
        """(role, dimension, measure, metric) for every score / metric comparison."""
        out = []
        for role, dims in DIMENSIONS_BY_ROLE.items():
            for dim in dims:
                for layer, speaker in TARGETS[dim]:
                    for metric in ASR_METRICS if layer == "transcription" else MT_METRICS:
                        out.append((role, dim, measure_name(layer, speaker, metric), metric))
        return out

    # --- correlations -----------------------------------------------------------------

    @staticmethod
    def _within_language_ranks(table, columns):
        """Ranks within each language scaled to (0, 1]: removes the differences between languages."""
        return table.groupby("langue")[columns].rank(pct=True)

    @staticmethod
    def _spearman(a, b):
        ok = a.notna() & b.notna()
        if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
            return np.nan, np.nan, int(ok.sum())
        rho, p = spearmanr(a[ok], b[ok])
        return rho, p, int(ok.sum())

    def _correlate(self, table, within):
        rows = []
        pairs = self.pairs()
        columns = sorted({f"{r}_{d}" for r, d, _, _ in pairs} | {m for _, _, m, _ in pairs})
        data = self._within_language_ranks(table, columns) if within else table[columns]
        for role, dim, measure, metric in pairs:
            rho, p, n = self._spearman(data[f"{role}_{dim}"], data[measure])
            rows.append({"role": ROLE_NAMES[role], "dimension": DIMENSIONS[dim], "measure": measure,
                         "metric": metric, "n": n, "rho": rho, "p": p,
                         # oriented so that > 0 always means "the score follows the measured quality"
                         "agreement": rho if metric in HIGHER_IS_BETTER else -rho})
        out = pd.DataFrame(rows)
        ok = out.p.notna()
        out.loc[ok, "p_holm"] = holm(out.loc[ok, "p"])
        return out

    def correlate(self):
        pooled = self._correlate(self.table, within=False).assign(analysis="pooled")
        within = self._correlate(self.table, within=True).assign(analysis="within_language")
        self.correlations = pd.concat([pooled, within], ignore_index=True)
        self.correlations_by_language = pd.concat(
            [self._correlate(g, within=False).assign(langue=lang) for lang, g in self.table.groupby("langue")],
            ignore_index=True)
        return self.correlations

    # --- translation scores vs. omissions ----------------------------------------------

    def omissions_and_translation(self):
        """Raters judge the translation of the displayed transcription, where omitted passages are invisible, whereas
        the metrics compare it with the translation of what was actually said. Two within-language tests:
          - effect: do transcription errors (deletions, substitutions) lower the measured translation quality?
          - partial: score / metric agreement after removing the effect of the deletions (partial Spearman: rank
            residuals on the deletion rate). If omissions were the only cause of the disagreement, it would rise.
        """
        rows = []
        ranks = self._within_language_ranks(self.table, list(self.table.columns[1:]))
        for speaker in ("caller", "operator"):
            for error in ("del_rate", "sub_rate"):
                for metric in MT_METRICS:
                    a, b = f"asr_{error}_{speaker}", f"mt_{metric}_{speaker}"
                    rho, p, n = self._spearman(ranks[a], ranks[b])
                    rows.append({"test": "effect", "x": a, "y": b, "control": "", "n": n, "rho": rho, "p": p})
        scores = [("appelant", "traduction_vers_fr", "caller"), ("operateur", "traduction_vers_fr", "caller"),
                  ("appelant", "traduction_vers_appelant", "operator")]
        for role, dim, speaker in scores:
            control = f"asr_del_rate_{speaker}"
            for metric in MT_METRICS:
                score, measure = f"{role}_{dim}", f"mt_{metric}_{speaker}"
                sign = 1 if metric in HIGHER_IS_BETTER else -1
                for ctrl in ("", control):
                    x, y = ranks[score], ranks[measure]
                    if ctrl:
                        x, y = self._residuals(x, ranks[ctrl]), self._residuals(y, ranks[ctrl])
                    rho, p, n = self._spearman(x, y)
                    rows.append({"test": "partial" if ctrl else "raw", "x": f"{ROLE_NAMES[role]}_{DIMENSIONS[dim]}",
                                 "y": measure, "control": ctrl, "n": n, "rho": sign * rho, "p": p})
        out = pd.DataFrame(rows)
        out["p_holm"] = holm(out.p)
        self.omissions = out
        return out

    @staticmethod
    def _residuals(y, x):
        ok = y.notna() & x.notna()
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        return y - (slope * x + intercept)

    def score_distribution(self):
        cols = [f"{r}_{d}" for r, dims in DIMENSIONS_BY_ROLE.items() for d in dims]
        dist = self.table[cols].apply(lambda c: c.value_counts()).reindex(range(1, 6)).fillna(0).astype(int)
        dist.columns = [f"{ROLE_NAMES[c.split('_', 1)[0]]}_{DIMENSIONS[c.split('_', 1)[1]]}" for c in cols]
        dist.index.name = "score"
        self.distribution = dist
        return dist

    # --- per language -----------------------------------------------------------------

    def summarize(self):
        """Mean score per language and role, next to the mean measured metric of the matching texts."""
        rows = []
        for lang, g in list(self.table.groupby("langue")) + [("TOTAL", self.table)]:
            row = {"langue": lang, "calls": len(g)}
            for role, dims in DIMENSIONS_BY_ROLE.items():
                for dim in dims:
                    row[f"{ROLE_NAMES[role]}_{DIMENSIONS[dim]}"] = g[f"{role}_{dim}"].mean()
            for col in [c for c in self.table.columns if c.startswith(("asr_", "mt_"))]:
                row[col] = g[col].mean()
            rows.append(row)
        self.per_language = pd.DataFrame(rows)
        return self.per_language

    # --- figures ----------------------------------------------------------------------

    def plot_correlations(self):
        """Heatmap of the within-language agreement between each score and each matching measure."""
        c = self.correlations[self.correlations.analysis == "within_language"].copy()
        c["row"] = c.role + " | " + c.dimension
        c["col"] = c.measure
        rows = list(dict.fromkeys(c.row))
        cols = list(dict.fromkeys(c.col))
        grid = c.pivot(index="row", columns="col", values="agreement").reindex(index=rows, columns=cols)
        sig = c.pivot(index="row", columns="col", values="p_holm").reindex(index=rows, columns=cols)

        cmap = LinearSegmentedColormap.from_list("rdbu", ["#2166ac", "#f7f7f7", "#b2182b"])
        fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 3, 0.45 * len(rows) + 2))
        ax.imshow(np.ma.masked_invalid(grid.values), cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = grid.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}{stars(sig.values[i, j])}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        ax.set_title("Agreement between questionnaire scores and measured quality (within language, n = "
                     f"{len(self.table)})\n> 0: the score is higher when the measured quality is better; "
                     "Holm-corrected", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"correlations.{ext}", dpi=200)
        plt.close(fig)

    def plot_scores(self):
        """Measured semantic error by score level: transcription (whole call) and translation into French."""
        panels = [("transcription", "asr_semdist_call", "Transcription score", "SemDist of the transcription"),
                  ("traduction_vers_fr", "mt_semdist_caller", "Translation into French score",
                   "SemDist of the translation (caller -> FR)")]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
        colors = {"appelant": "#1f77b4", "operateur": "#ff7f0e"}
        for ax, (dim, measure, xlabel, ylabel) in zip(axes, panels):
            for k, role in enumerate(DIMENSIONS_BY_ROLE):
                data = [self.table.loc[self.table[f"{role}_{dim}"] == s, measure].dropna() for s in range(1, 6)]
                pos = np.arange(1, 6) + (k - 0.5) * 0.35
                keep = [i for i, d in enumerate(data) if len(d)]
                bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=0.3, patch_artist=True,
                                showfliers=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor(colors[role])
                    patch.set_alpha(0.6)
                for i in keep:
                    ax.text(pos[i], ax.get_ylim()[0], f"{len(data[i])}", ha="center", va="bottom", fontsize=6)
                ax.plot([], [], color=colors[role], lw=6, alpha=0.6, label=ROLE_NAMES[role])
            ax.set_xticks(range(1, 6), range(1, 6))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.suptitle("Measured error by questionnaire score (lower = better; numbers = calls)", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"scores_vs_semdist.{ext}", dpi=200)
        plt.close(fig)

    # --- output -----------------------------------------------------------------------

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        kw = {"index": False, "encoding": "utf-8-sig"}
        self.correlations.to_csv(OUT_DIR / "correlations.csv", **kw)
        self.correlations_by_language.to_csv(OUT_DIR / "correlations_by_language.csv", **kw)
        self.per_language.to_csv(OUT_DIR / "per_language.csv", **kw)
        self.omissions.to_csv(OUT_DIR / "omissions_and_translation.csv", **kw)
        self.distribution.to_csv(OUT_DIR / "score_distribution.csv", encoding="utf-8-sig")
        self.plot_correlations()
        self.plot_scores()
        print(f"-> {OUT_DIR}")

    def run(self):
        self.correlate()
        self.summarize()
        self.omissions_and_translation()
        self.score_distribution()
        self.save()
        return self


def main():
    rq = PerceivedVsMeasured().run()
    pd.set_option("display.width", 220, "display.max_rows", 200, "display.float_format", "{:.3f}".format)
    cols = ["role", "dimension", "measure", "n", "rho", "agreement", "p_holm"]
    for analysis, g in rq.correlations.groupby("analysis", sort=False):
        print(f"\n=== {analysis}\n{g[cols].to_string(index=False)}")
    print(f"\n=== translation scores vs. omissions (within language)\n{rq.omissions.to_string(index=False)}")
    print(f"\n=== score distribution\n{rq.distribution.to_string()}")


if __name__ == "__main__":
    main()
