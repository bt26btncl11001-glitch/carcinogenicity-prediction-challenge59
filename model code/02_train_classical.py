"""Step 2: 12 classical combinations, 5x5 random and scaffold CV (pooled out-of-fold),
fixed holdouts with bootstrap CI, saved checkpoints."""
import pickle, joblib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import matthews_corrcoef
from common import *

df, mols, y = load_data()
XS = {k: f(mols) for k, f in FEATS.items()}
cvs = {"random": load_cv("cv_random_5x5"), "scaffold": load_cv("cv_scaffold_5x5")}


def run_fold(sp, r, f, m):
    tr, te = np.array(r["train"]), np.array(r["test"])
    s, thr, _ = fit_score(m, f, XS[f][tr], y[tr], XS[f][te], SEED + r["repeat"])
    return sp, f, m, r["repeat"], te, s, thr


jobs = [(sp, r, f, m) for sp, cv in cvs.items() for r in cv for f in FEATS for m in MODELS]
print("fits:", len(jobs))
out = Parallel(n_jobs=-1, verbose=1)(delayed(run_fold)(*j) for j in jobs)

pool = {}
for sp, f, m, rep, te, s, thr in out:
    d = pool.setdefault((sp, f, m, rep), {"idx": [], "s": [], "thr": thr})
    d["idx"] += te.tolist(); d["s"] += s.tolist()
rows = []
for (sp, f, m, rep), d in pool.items():
    idx, s = np.array(d["idx"]), np.array(d["s"])
    rows.append(dict(split=sp, features=f, model=m, repeat=rep, **metrics(y[idx], s, d["thr"])))
res = pd.DataFrame(rows)
res.to_csv(REPO / "results" / "s3_cv_metrics_per_repeat.csv", index=False)
pickle.dump(pool, open(REPO / "results" / "s3_oof_predictions.pkl", "wb"))
summ = res.groupby(["split", "features", "model"])[["MCC", "AUROC", "AUPRC", "BalAcc"]].agg(["mean", "std"]).round(3)
summ.to_csv(REPO / "results" / "s3_cv_summary.csv")
print(summ)


def boot_ci(yt, s, thr, n=1000):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() != yt[i].max():
            v.append(matthews_corrcoef(yt[i], (s[i] >= thr).astype(int)))
    return np.percentile(v, [2.5, 97.5])


rows = []
for col, tag in [("scaffold_split", "scaffold"), ("random_split", "random")]:
    tr = np.where(df[col] == "train")[0]; te = np.where(df[col] == "test")[0]
    for f in FEATS:
        for m in MODELS:
            s, thr, mod = fit_score(m, f, XS[f][tr], y[tr], XS[f][te], SEED)
            lo, hi = boot_ci(y[te], s, thr)
            rows.append(dict(split=tag, features=f, model=m, MCC_lo=lo, MCC_hi=hi, **metrics(y[te], s, thr)))
            if tag == "scaffold":
                joblib.dump(mod, REPO / "checkpoints" / f"holdout_scaffold_{f}_{m}.joblib", compress=3)
pd.DataFrame(rows).round(3).to_csv(REPO / "results" / "s3_holdout_metrics.csv", index=False)

# final models on ALL data for the top-3 scaffold-CV combinations (for the screening tool only)
top3 = (res[res.split == "scaffold"].groupby(["features", "model"]).MCC.mean()
        .sort_values(ascending=False).head(3).index)
for f, m in top3:
    _, _, mod = fit_score(m, f, XS[f], y, XS[f][:5], SEED)
    joblib.dump(mod, REPO / "checkpoints" / f"final_{f}_{m}.joblib", compress=3)
print("done; final models:", list(top3))
