"""Step 1: load TDC Carcinogens_Lagunin, curate, audit, and build every split."""
import json
import numpy as np, pandas as pd
from tdc.single_pred import Tox
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold, RepeatedStratifiedKFold, train_test_split
from common import *

data = Tox(name="Carcinogens_Lagunin", path=str(REPO / "data" / "tdc_raw"))
raw = data.get_data()
raw.to_csv(REPO / "data" / "raw_carcinogens_lagunin.csv", index=False)
official = data.get_split()
for k, v in official.items():
    v.to_csv(REPO / "splits" / f"tdc_official_{k}.csv", index=False)

# ---- curation ----
df = raw.copy()
log = {"n_raw": len(df)}
df["smiles_can"] = df["Drug"].apply(standardize)
log["n_invalid_or_inorganic"] = int(df["smiles_can"].isna().sum())
df = df.dropna(subset=["smiles_can"])
g = df.groupby("smiles_can")["Y"].agg(["nunique", "count"])
conflicts = g[g["nunique"] > 1].index
consistent = g[g["nunique"] == 1]
log["n_conflicting_label_smiles"] = int(len(conflicts))
log["n_consistent_duplicate_rows_removed"] = int((consistent["count"] - 1).sum())
df = df[~df["smiles_can"].isin(conflicts)].drop_duplicates("smiles_can").reset_index(drop=True)
df = df.rename(columns={"Drug_ID": "name", "Y": "label"})[["name", "smiles_can", "label"]]
log.update(n_final=len(df), n_carcinogen=int(df.label.sum()),
           n_non_carcinogen=int((df.label == 0).sum()), carcinogen_fraction=round(float(df.label.mean()), 4))
sm = {k: set(v["Drug"].apply(standardize).dropna()) for k, v in official.items()}
log["official_overlap_train_test"] = len(sm["train"] & sm["test"])
log["official_overlap_train_valid"] = len(sm["train"] & sm["valid"])
log["official_overlap_valid_test"] = len(sm["valid"] & sm["test"])
df.to_csv(REPO / "data" / "curated_carcinogens.csv", index=False)
json.dump(log, open(REPO / "results" / "s1_curation_log.json", "w"), indent=2)
print(json.dumps(log, indent=2))

# ---- scaffolds and groups ----
mols = [Chem.MolFromSmiles(s) for s in df.smiles_can]
y = df.label.values
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
df["group"] = [s if s != "" else f"ACYC_{i}" for i, s in enumerate(scaf)]
groups = df["group"].values
idx_all = np.arange(len(df))


def recs(pairs):
    return [{"repeat": k // 5, "fold": k % 5, "train": tr.tolist(), "test": te.tolist()}
            for k, (tr, te) in enumerate(pairs)]


# ---- random and scaffold CV ----
rand_cv = recs(list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(idx_all, y)))
pairs = []
for rep in range(5):
    pairs += list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep).split(idx_all, y, groups))
scaf_cv = recs(pairs)

# ---- cluster CV (Butina, Tanimoto >= 0.4) ----
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(m) for m in mols]
dists = []
for i in range(1, len(fps)):
    dists.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
cid = np.zeros(len(fps), dtype=int)
for k, c in enumerate(Butina.ClusterData(dists, len(fps), 0.6, isDistData=True)):
    for i in c:
        cid[i] = k
pairs = []
for rep in range(5):
    pairs += list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep).split(idx_all, y, cid))
clus_cv = recs(pairs)

for name, cv in [("cv_random_5x5", rand_cv), ("cv_scaffold_5x5", scaf_cv), ("cv_cluster_5x5", clus_cv)]:
    json.dump(cv, open(REPO / "splits" / f"{name}.json", "w"))

# ---- fixed holdouts ----
trv, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(idx_all, y, groups))
a, b = next(StratifiedGroupKFold(7, shuffle=True, random_state=SEED).split(trv, y[trv], groups[trv]))
tr, va = trv[a], trv[b]
r_trv, r_te = train_test_split(idx_all, test_size=len(te), stratify=y, random_state=SEED)
r_tr, r_va = train_test_split(r_trv, test_size=len(va), stratify=y[r_trv], random_state=SEED)
df["scaffold_split"] = ""
df.loc[tr, "scaffold_split"] = "train"; df.loc[va, "scaffold_split"] = "valid"; df.loc[te, "scaffold_split"] = "test"
df["random_split"] = ""
df.loc[r_tr, "random_split"] = "train"; df.loc[r_va, "random_split"] = "valid"; df.loc[r_te, "random_split"] = "test"
df[["name", "smiles_can", "label", "group", "scaffold_split", "random_split"]].to_csv(
    REPO / "splits" / "holdout_splits.csv", index=False)

# ---- leakage audit ----
for cv, by_group, gr in [(rand_cv, False, None), (scaf_cv, True, groups), (clus_cv, True, cid)]:
    for r in cv:
        a_, b_ = np.array(r["train"]), np.array(r["test"])
        assert not set(df.smiles_can[a_]) & set(df.smiles_can[b_]), "molecule leakage"
        if by_group:
            assert not set(gr[a_]) & set(gr[b_]), "group leakage"
for col in ["scaffold_split", "random_split"]:
    assert not set(df[df[col] == "train"].smiles_can) & set(df[df[col] == "test"].smiles_can)
print("Leakage audit PASSED")
