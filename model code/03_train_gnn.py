"""Step 3: small GIN ensemble, evaluated on the identical random and scaffold CV splits (resumable)."""
import pickle, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from common import *

torch.set_num_threads(2)
device = "cuda" if torch.cuda.is_available() else "cpu"
df, mols, y = load_data()

ATOMS = [6, 7, 8, 16, 15, 9, 17, 35, 53]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]


def onehot(v, ch):
    o = [0.0] * (len(ch) + 1)
    o[ch.index(v) if v in ch else len(ch)] = 1.0
    return o


def atom_feat(a):
    return (onehot(a.GetAtomicNum(), ATOMS) + onehot(a.GetTotalDegree(), [1, 2, 3, 4]) +
            onehot(a.GetFormalCharge(), [-1, 0, 1]) + onehot(a.GetHybridization(), HYB) +
            onehot(a.GetTotalNumHs(), [0, 1, 2, 3]) + [float(a.GetIsAromatic()), float(a.IsInRing())])


def mol_graph(m):
    x = torch.tensor([atom_feat(a) for a in m.GetAtoms()], dtype=torch.float32)
    src, dst = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); src += [i, j]; dst += [j, i]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return x, ei


GRAPHS = [mol_graph(m) for m in mols]
N_FEAT = GRAPHS[0][0].shape[1]


def collate(idx):
    xs, eis, bs, off = [], [], [], 0
    for k, i in enumerate(idx):
        x, ei = GRAPHS[i]
        xs.append(x); eis.append(ei + off); bs.append(torch.full((x.shape[0],), k, dtype=torch.long)); off += x.shape[0]
    return torch.cat(xs).to(device), torch.cat(eis, 1).to(device), torch.cat(bs).to(device), len(idx)


class GINLayer(nn.Module):
    def __init__(s, d, drop):
        super().__init__()
        s.eps = nn.Parameter(torch.zeros(1))
        s.mlp = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d))
        s.bn, s.drop = nn.BatchNorm1d(d), nn.Dropout(drop)

    def forward(s, h, ei):
        agg = torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]])
        return s.drop(F.relu(s.bn(s.mlp((1 + s.eps) * h + agg)))) + h


class GNN(nn.Module):
    def __init__(s, din, d=64, L=3, drop=0.2):
        super().__init__()
        s.inp = nn.Linear(din, d)
        s.layers = nn.ModuleList([GINLayer(d, drop) for _ in range(L)])
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(s, x, ei, batch, B):
        h = s.inp(x)
        for l in s.layers:
            h = l(h, ei)
        sm = torch.zeros(B, h.shape[1], device=h.device).index_add_(0, batch, h)
        cnt = torch.zeros(B, device=h.device).index_add_(0, batch, torch.ones(h.shape[0], device=h.device)).clamp(min=1).unsqueeze(1)
        return s.head(torch.cat([sm / cnt, sm], 1)).squeeze(1)


def train_gnn(tr_idx, seed, epochs=100, bs=32, lr=2e-3, wd=1e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GNN(N_FEAT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ytr = y[tr_idx]
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for k in range(0, len(perm), bs):
            b = perm[k:k + bs]
            if len(b) < 2:
                continue
            x, ei, bt, B = collate(b)
            yt = torch.tensor(y[b], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(model(x, ei, bt, B), yt, pos_weight=pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict_gnn(model, idx):
    model.eval(); out = []
    for k in range(0, len(idx), 128):
        x, ei, bt, B = collate(idx[k:k + 128])
        out.append(torch.sigmoid(model(x, ei, bt, B)).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    N_SEEDS = 2
    path = REPO / "results" / "s4_gnn_oof.pkl"
    done = pickle.load(open(path, "rb")) if path.exists() else {}
    cvs = {"scaffold": load_cv("cv_scaffold_5x5"), "random": load_cv("cv_random_5x5")}
    t0 = time.time()
    for sp, cv in cvs.items():
        for rep in range(5):
            if (sp, rep) in done:
                continue
            idx_all, s_all = [], []
            for r in [r for r in cv if r["repeat"] == rep]:
                tr, te = np.array(r["train"]), np.array(r["test"])
                ps = [predict_gnn(train_gnn(tr, SEED + rep * 100 + k), te) for k in range(N_SEEDS)]
                idx_all += te.tolist(); s_all += np.mean(ps, axis=0).tolist()
            done[(sp, rep)] = {"idx": idx_all, "s": s_all}
            pickle.dump(done, open(path, "wb"))
            print(sp, rep, f"{(time.time() - t0) / 60:.1f} min")
    rows = []
    for (sp, rep), d in done.items():
        idx, s = np.array(d["idx"]), np.array(d["s"])
        rows.append(dict(split=sp, features="Graph", model="GIN-ensemble", repeat=rep, **metrics(y[idx], s, 0.5)))
    g = pd.DataFrame(rows)
    g.to_csv(REPO / "results" / "s4_gnn_cv_metrics_per_repeat.csv", index=False)
    print(g.groupby("split")[["MCC", "AUROC", "BalAcc"]].agg(["mean", "std"]).round(3))
    for k in range(5):   # final ensemble on all data (for reference only)
        torch.save(train_gnn(np.arange(len(y)), SEED + k).state_dict(), REPO / "checkpoints" / f"final_GNN_seed{k}.pt")
