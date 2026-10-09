"""Research-use carcinogenicity screening: MACCS keys + Random Forest trained on 274 curated
TDC Carcinogens_Lagunin molecules.  Educational / research use only; not for regulatory or clinical decisions.

Usage:
  python predict.py --smiles "CCNN" "c1ccccc1N"
  python predict.py --csv input.csv --col smiles --out predictions.csv
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, joblib
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "final_MACCS_RandomForest.joblib"
TRAIN_PATH = ROOT / "data" / "curated_carcinogens.csv"
SIM_LOW, SIM_HIGH = 0.24, 0.35   # similarity tertile boundaries from cross-validation (Section 8)

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def standardize(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    try:
        m = _uncharger.uncharge(_chooser.choose(m))
        Chem.SanitizeMol(m)
        if not any(a.GetSymbol() == "C" for a in m.GetAtoms()):
            return None
        return Chem.MolToSmiles(m, isomericSmiles=False)
    except Exception:
        return None


def predict(smiles_list):
    model = joblib.load(MODEL_PATH)
    tr = pd.read_csv(TRAIN_PATH)
    train_set = set(tr.smiles_can)
    train_fps = [_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.smiles_can]
    rows = []
    for raw in smiles_list:
        can = standardize(raw)
        if can is None:
            rows.append(dict(input=raw, canonical=None, p_carcinogen=np.nan, call="invalid input",
                             nearest_train_similarity=np.nan, reliability="n/a",
                             note="could not parse, or not an organic molecule"))
            continue
        m = Chem.MolFromSmiles(can)
        x = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32).reshape(1, -1)
        p = float(model.predict_proba(x)[0, 1])
        sim = float(max(DataStructs.BulkTanimotoSimilarity(_gen.GetFingerprint(m), train_fps)))
        rel = "higher" if sim >= SIM_HIGH else ("medium" if sim >= SIM_LOW else "lower")
        note = "molecule is in the training set; prediction is not independent" if can in train_set else ""
        rows.append(dict(input=raw, canonical=can, p_carcinogen=round(p, 3),
                         call="carcinogen" if p >= 0.5 else "non-carcinogen",
                         nearest_train_similarity=round(sim, 3), reliability=rel, note=note))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", nargs="*", default=[])
    ap.add_argument("--csv"); ap.add_argument("--col", default="smiles"); ap.add_argument("--out")
    a = ap.parse_args()
    smi = list(a.smiles)
    if a.csv:
        smi += pd.read_csv(a.csv)[a.col].tolist()
    res = predict(smi)
    print(res.to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)
    print("\nResearch use only. Reliability is lower when the nearest training molecule is dissimilar.")
