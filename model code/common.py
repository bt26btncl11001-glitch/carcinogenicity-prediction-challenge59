"""Shared helpers: paths, seeds, standardization, features, models, metrics."""
import json, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
                             average_precision_score)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
for _d in ["data", "splits", "checkpoints", "results", "figures"]:
    (REPO / _d).mkdir(exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
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


_mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def feat_morgan(ms):
    return np.vstack([_mg.GetCountFingerprintAsNumPy(m) for m in ms]).astype(np.float32)


def feat_maccs(ms):
    return np.vstack([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in ms])


def _safe(f, m):
    try:
        return f(m)
    except Exception:
        return 0.0


def feat_desc(ms):
    X = np.array([[_safe(f, m) for _, f in Descriptors.descList] for m in ms], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)


FEATS = {"Morgan": feat_morgan, "MACCS": feat_maccs, "RDKitDesc": feat_desc}
MODELS = ["LogReg", "SVM", "RandomForest", "XGBoost"]


def make_model(name, feat, ytr, seed):
    pre = [StandardScaler()] if feat != "MACCS" else []
    if name == "LogReg":
        return make_pipeline(*pre, LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    if name == "SVM":
        return make_pipeline(*pre, SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced"))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                      n_jobs=1, random_state=seed)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.5, scale_pos_weight=spw, eval_metric="logloss",
                         tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)


def fit_score(name, feat, Xtr, ytr, Xte, seed):
    m = make_model(name, feat, ytr, seed).fit(Xtr, ytr)
    if name == "SVM":
        return m.decision_function(Xte), 0.0, m
    return m.predict_proba(Xte)[:, 1], 0.5, m


def metrics(yt, s, thr):
    yp = (s >= thr).astype(int)
    return dict(MCC=matthews_corrcoef(yt, yp), AUROC=roc_auc_score(yt, s),
                AUPRC=average_precision_score(yt, s), BalAcc=balanced_accuracy_score(yt, yp))


def load_data():
    df = pd.read_csv(REPO / "splits" / "holdout_splits.csv")
    mols = [Chem.MolFromSmiles(s) for s in df.smiles_can]
    return df, mols, df.label.values


def load_cv(name):
    return json.load(open(REPO / "splits" / f"{name}.json"))
