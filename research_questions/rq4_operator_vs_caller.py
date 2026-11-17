"""RQ4 - Operator vs. caller: can the measured errors explain the gap between operator and caller scores?

Study 1 found that the operator rates the system lower than the callers on every dimension but the transcription.
Two hypotheses:
  H1  the operator applies stricter criteria (professional awareness of the consequences of a misunderstanding);
  H2  the bilingual callers can check the translation against what they said, whereas the monolingual operator
      depends entirely on the system output.

  1. gap          Wilcoxon signed-rank test per shared dimension, operator vs caller on the same call (77 calls),
                  and per language (Holm-corrected).
  2. H1           both rate the translation into French of the same text: a gap there cannot come from the quality.
                  Gap vs measured quality (within-language ranks): does the operator penalise errors more?
                  Agreement of each rater with the measured quality (as in RQ2), difference by stratified bootstrap.
  3. H2           fidelity of the automatic translation to the displayed automatic transcription (cross-lingual
                  SemDist, BGE-M3): only the caller can read both. H2 predicts that the caller's translation score
                  follows the fidelity more than the operator's, and that the gap (caller - operator) grows with the
                  fidelity (a faithful translation is recognised as such by the caller only).

All correlations are within language (one caller per language), Spearman, Holm-corrected within each family.

Input: resultats/rq1/metrics_per_call.csv (run rq1_metrics first).
Outputs in resultats/rq4/:
  - gap.csv                     Wilcoxon tests, all calls and per language
  - h1_gap_vs_quality.csv       gap vs measured quality
  - agreement_difference.csv    operator - caller agreement with the measured quality (bootstrap CI)
  - fidelity_per_call.csv       cross-lingual SemDist, automatic and manual (baseline), per call and speaker
  - h2_fidelity.csv             scores and gap vs fidelity
  - gap.png / .pdf              mean gap per dimension and language

Run:  python -m research_questions.rq4_operator_vs_caller
"""

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from corpus import DIMENSIONS
from research_questions.common import RESULTS_DIR, holm, split_speakers, stars
from research_questions.rq2_perceived_vs_measured import PerceivedVsMeasured

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = RESULTS_DIR / "rq4"
SEMDIST_MODEL = "BAAI/bge-m3"
SHARED = ["transcription", "traduction_vers_fr", "compris", "confiance"]  # rated by both
LANGUAGE_NAMES = {"anglais": "English", "arabe": "Arabic", "espagnol": "Spanish", "italien": "Italian"}
ORDER = ["arabe", "anglais", "italien", "espagnol"]
# measured quality each dimension refers to (whole call, or the translated caller turns); lower = better
QUALITY = {"transcription": "asr_semdist_call", "traduction_vers_fr": "mt_semdist_caller",
           "compris": "mt_semdist_call", "confiance": "mt_semdist_call"}
N_BOOTSTRAP = 2000
SEED = 0


class OperatorVsCaller:
    """Test the operator-caller gap and the hypotheses H1 and H2."""

    def __init__(self, device=None):
        self.device = device
        base = PerceivedVsMeasured()
        self.corpus = base.corpus
        self.table = base.table.copy()
        for dim in SHARED:
            self.table[f"gap_{dim}"] = self.table[f"operateur_{dim}"] - self.table[f"appelant_{dim}"]

    # --- helpers ----------------------------------------------------------------------

    def _ranks(self, columns, table=None):
        table = self.table if table is None else table
        return table.groupby("langue")[columns].rank(pct=True)

    @staticmethod
    def _spearman(a, b):
        ok = a.notna() & b.notna()
        rho, p = spearmanr(a[ok], b[ok])
        return rho, p, int(ok.sum())

    # --- 1. gap -----------------------------------------------------------------------

    def gap(self):
        rows = []
        for lang in [None] + ORDER:
            g = self.table if lang is None else self.table[self.table.langue == lang]
            for dim in SHARED:
                op, ca = g[f"operateur_{dim}"], g[f"appelant_{dim}"]
                diff = op - ca
                stat, p = wilcoxon(op, ca) if (diff != 0).any() else (np.nan, 1.0)
                rows.append({"langue": lang or "ALL", "dimension": DIMENSIONS[dim], "n": len(g),
                             "mean_operator": op.mean(), "mean_caller": ca.mean(), "mean_gap": diff.mean(),
                             "operator_lower": (diff < 0).sum(), "equal": (diff == 0).sum(),
                             "operator_higher": (diff > 0).sum(), "W": stat, "p": p})
        out = pd.DataFrame(rows)
        overall = out.langue == "ALL"
        out.loc[overall, "p_holm"] = holm(out.loc[overall, "p"])
        out.loc[~overall, "p_holm"] = holm(out.loc[~overall, "p"])
        self.gaps = out
        return out

    # --- 2. H1 ------------------------------------------------------------------------

    def h1(self):
        """Gap vs measured quality, and difference of agreement between the two raters."""
        rows = []
        for dim in SHARED:
            q = QUALITY[dim]
            r = self._ranks([f"gap_{dim}", q])
            rho, p, n = self._spearman(r[f"gap_{dim}"], r[q])
            # q is an error (lower = better): rho > 0 means the gap grows (operator relatively higher) with errors
            rows.append({"dimension": DIMENSIONS[dim], "quality": q, "n": n, "rho_gap_vs_error": rho, "p": p})
        self.h1_gap = pd.DataFrame(rows)
        self.h1_gap["p_holm"] = holm(self.h1_gap.p)

        rng = np.random.default_rng(SEED)
        rows = []
        for dim in SHARED:
            q = QUALITY[dim]
            cols = [f"operateur_{dim}", f"appelant_{dim}", q]

            def agreement(table):
                r = self._ranks(cols, table)
                op = -spearmanr(r[cols[0]], r[q])[0]
                ca = -spearmanr(r[cols[1]], r[q])[0]
                return op, ca

            op, ca = agreement(self.table)
            boot = []
            groups = [g for _, g in self.table.groupby("langue")]
            for _ in range(N_BOOTSTRAP):
                sample = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
                o, c = agreement(sample)
                boot.append(o - c)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            rows.append({"dimension": DIMENSIONS[dim], "quality": q, "agreement_operator": op,
                         "agreement_caller": ca, "difference": op - ca, "ci_low": lo, "ci_high": hi})
        self.agreement = pd.DataFrame(rows)
        return self.h1_gap

    # --- 3. H2 ------------------------------------------------------------------------

    def fidelity(self):
        """Cross-lingual SemDist between each speaker's transcription and its translation (automatic: what the
        screen showed; manual: baseline of a correct translation, for the scale of each language pair)."""
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        rows = []
        for r in self.corpus.itertuples():
            texts = {"auto": (split_speakers(r.transcription_auto), split_speakers(r.traduction_auto)),
                     "manual": (split_speakers(r.transcription_manuelle), split_speakers(r.traduction_manuelle))}
            for version, (src, tgt) in texts.items():
                for speaker in ("Caller", "Operator"):
                    rows.append({"call": r.call, "langue": r.langue, "version": version, "speaker": speaker,
                                 "src": src[speaker], "tgt": tgt[speaker]})
        f = pd.DataFrame(rows)
        enc = SentenceTransformer(SEMDIST_MODEL, device=device)
        e_src = enc.encode(list(f.src), normalize_embeddings=True, convert_to_tensor=True, batch_size=16)
        e_tgt = enc.encode(list(f.tgt), normalize_embeddings=True, convert_to_tensor=True, batch_size=16)
        f["cross_semdist"] = (1 - (e_src * e_tgt).sum(dim=1)).cpu().numpy()
        f.loc[(f.src.str.strip() == "") | (f.tgt.str.strip() == ""), "cross_semdist"] = np.nan
        self.fidelity_calls = f.drop(columns=["src", "tgt"])
        wide = self.fidelity_calls[self.fidelity_calls.version == "auto"].pivot(
            index="call", columns="speaker", values="cross_semdist")
        self.table["fidelity_caller"] = wide["Caller"]
        self.table["fidelity_operator"] = wide["Operator"]
        return self.fidelity_calls

    def h2(self):
        rows = []
        tests = [("operateur_traduction_vers_fr", "fidelity_caller"), ("appelant_traduction_vers_fr", "fidelity_caller"),
                 ("gap_traduction_vers_fr", "fidelity_caller"),
                 ("appelant_traduction_vers_appelant", "fidelity_operator")]
        for score, fid in tests:
            r = self._ranks([score, fid])
            rho, p, n = self._spearman(r[score], r[fid])
            # fidelity is a distance: agreement = -rho (> 0: higher score when the translation is more faithful)
            rows.append({"score": score, "fidelity": fid, "n": n, "rho": rho, "agreement": -rho, "p": p})
        out = pd.DataFrame(rows)
        out["p_holm"] = holm(out.p)
        self.h2_tests = out
        return out

    # --- figure -----------------------------------------------------------------------

    def plot_gap(self):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        width = 0.18
        labels = [DIMENSIONS[d] for d in SHARED]
        for k, lang in enumerate(["ALL"] + ORDER):
            g = self.gaps[self.gaps.langue == lang].set_index("dimension").reindex(labels)
            x = np.arange(len(SHARED)) + (k - 2) * width
            ax.bar(x, g.mean_gap, width, label="All calls" if lang == "ALL" else LANGUAGE_NAMES[lang],
                   color="#444444" if lang == "ALL" else None)
            for xi, v, p in zip(x, g.mean_gap, g.p_holm):
                if stars(p):
                    ax.text(xi, v - 0.06 if v < 0 else v + 0.02, stars(p), ha="center",
                            va="top" if v < 0 else "bottom", fontsize=7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(self.gaps.mean_gap.min() - 0.2, self.gaps.mean_gap.max() + 0.15)
        ax.set_xticks(range(len(SHARED)), [l.replace("_", " ") for l in labels], fontsize=8)
        ax.set_ylabel("Mean gap (operator - caller)")
        ax.set_title("Operator - caller score gap (Wilcoxon signed-rank, Holm-corrected)", fontsize=9)
        ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"gap.{ext}", dpi=200)
        plt.close(fig)

    # --- output -----------------------------------------------------------------------

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        kw = {"index": False, "encoding": "utf-8-sig"}
        self.gaps.to_csv(OUT_DIR / "gap.csv", **kw)
        self.h1_gap.to_csv(OUT_DIR / "h1_gap_vs_quality.csv", **kw)
        self.agreement.to_csv(OUT_DIR / "agreement_difference.csv", **kw)
        self.fidelity_calls.to_csv(OUT_DIR / "fidelity_per_call.csv", **kw)
        self.h2_tests.to_csv(OUT_DIR / "h2_fidelity.csv", **kw)
        self.plot_gap()
        print(f"-> {OUT_DIR}")

    def run(self):
        self.gap()
        self.h1()
        self.fidelity()
        self.h2()
        self.save()
        return self


def main():
    rq = OperatorVsCaller().run()
    pd.set_option("display.width", 220, "display.max_rows", 100, "display.float_format", "{:.3f}".format)
    print(f"\n=== gap (operator - caller)\n{rq.gaps.to_string(index=False)}")
    print(f"\n=== H1: gap vs measured error (within language)\n{rq.h1_gap.to_string(index=False)}")
    print(f"\n=== H1: agreement with the measured quality\n{rq.agreement.to_string(index=False)}")
    fid = rq.fidelity_calls.groupby(["version", "speaker", "langue"]).cross_semdist.mean().unstack(0)
    print(f"\n=== fidelity (cross-lingual SemDist, mean)\n{fid.to_string()}")
    print(f"\n=== H2\n{rq.h2_tests.to_string(index=False)}")


if __name__ == "__main__":
    main()
