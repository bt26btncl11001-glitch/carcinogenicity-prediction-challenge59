"""Step 4: structural-alert enrichment + label-shuffling sanity test."""
import pickle
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score
from common import *

df, mols, y = load_data()

# Alerts defined BEFORE looking at results. Verify against Ashby & Tennant (1988) and Benigni-Bossa (2008).
ALERTS = {
    "Aromatic amine": ["c[NX3;H2,H1;!$(N-C=O);!$(N-S=O)]"],
    "Aromatic nitro": ["c[N+](=O)[O-]"],
    "Aromatic nitroso / N-OH": ["c[NX2]=O", "c[NX3][OX2H]"],
    "N-nitroso": ["[#7]-[NX2]=O"],
    "Azo (aromatic)": ["cN=Nc"],
    "Triazene / diazo / azide": ["[#7]=[#7]-[#7]", "[N+]#N", "N=[N+]=[N-]"],
    "Hydrazine": ["[NX3;!$(N-C=O)][NX3;!$(N-C=O)]"],
    "Epoxide / aziridine": ["C1OC1", "C1[NX3]C1"],
    "Alkyl halide": ["[CX4][Cl,Br,I]"],
    "Benzylic / allylic halide": ["[CX4;$(C-c),$(C-C=C)][Cl,Br,I]"],
    "N- or S-mustard": ["[NX3,SX2][CH2][CH2][Cl,Br,I]"],
    "Halo-alkene": ["[CX3]=[CX3][Cl,Br,I]"],
    "Alkyl sulfonate / sulfate": ["[CX4]O[SX4](=O)(=O)"],
    "Michael acceptor (broad; pattern too general)": ["[CX3;!a]=[CX3;!a]-[CX3]=O"],
    "Aldehyde": ["[CX3H1](=O)[#6]"],
    "beta-Lactone": ["O=C1CCO1"],
    "PAH (3+ fused rings)": ["c1ccc2c(c1)ccc1ccccc12", "c1ccc2cc3ccccc3cc2c1"],
}
PATS = {k: [Chem.MolFromSmarts(s) for s in v] for k, v in ALERTS.items()}
names = list(ALERTS)
H = np.array([[int(any(m.HasSubstructMatch(p) for p in PATS[a])) for a in names] for m in mols])
pd.DataFrame(H, columns=names).assign(smiles=df.smiles_can, label=y).to_csv(REPO / "results" / "s6_alert_matrix.csv", index=False)

rows = []
for j, a in enumerate(names):
    n = int(H[:, j].sum()); c = int(y[H[:, j] == 1].sum())
    if n:
        tab = [[c, n - c], [y.sum() - c, (len(y) - y.sum()) - (n - c)]]
        rows.append(dict(alert=a, n=n, carcinogens=c, rate=round(c / n, 3),
                         p_one_sided=round(fisher_exact(tab, alternative="greater")[1], 4)))
et = pd.DataFrame(rows).sort_values("rate", ascending=False)
et.to_csv(REPO / "results" / "s6_alert_enrichment.csv", index=False)
print(et.to_string(index=False))
rule = (H.sum(1) > 0).astype(int)
print("alert-only rule: MCC", round(matthews_corrcoef(y, rule), 3), "AUROC", round(roc_auc_score(y, H.sum(1)), 3),
      "BalAcc", round(balanced_accuracy_score(y, rule), 3))

# ---- label-shuffling control (scaffold CV, top-3 combinations) ----
XS = {k: f(mols) for k, f in FEATS.items()}
cv = load_cv("cv_scaffold_5x5")
combos = [("MACCS", "RandomForest"), ("RDKitDesc", "SVM"), ("Morgan", "RandomForest")]
yperm = {p: np.random.default_rng(1000 + p).permutation(y) for p in range(3)}


def shuffled_fold(f, m, r, p):
    tr, te = np.array(r["train"]), np.array(r["test"])
    s, thr, _ = fit_score(m, f, XS[f][tr], yperm[p][tr], XS[f][te], SEED)
    return f, m, r["repeat"], p, te, s, thr


out = Parallel(n_jobs=-1)(delayed(shuffled_fold)(f, m, r, p) for (f, m) in combos for r in cv for p in range(3))
pool = {}
for f, m, rep, p, te, s, thr in out:
    d = pool.setdefault((f, m, rep, p), {"idx": [], "s": [], "thr": thr})
    d["idx"] += te.tolist(); d["s"] += s.tolist()
sh = pd.DataFrame([dict(combo=f"{f}+{m}", MCC_shuffled=matthews_corrcoef(
    yperm[p][np.array(d["idx"])], (np.array(d["s"]) >= d["thr"]).astype(int)))
    for (f, m, rep, p), d in pool.items()])
sh.to_csv(REPO / "results" / "s5_label_shuffle.csv", index=False)
print(sh.groupby("combo").MCC_shuffled.agg(["mean", "std", "max"]).round(3))
