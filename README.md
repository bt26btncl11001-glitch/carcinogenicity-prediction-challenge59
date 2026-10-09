# Carcinogenicity Prediction from Chemical Structure
Challenge 59 (Safety), AI in Drug Discovery | Tathya Oza (BT26BTNCL11001) | Research and educational use only

## Data
TDC Carcinogens_Lagunin. 280 rows received (the TDC page lists 278); 274 unique molecules after
curation (largest fragment, uncharge, canonical non-stereo SMILES; 2 conflicting-label structures
removed; 2 duplicate rows collapsed). 58 carcinogens, 216 non-carcinogens (21.2%).
TDC's official split shares molecules across sets (1 train/test, 3 train/valid after standardization);
our splits were built after de-duplication and pass a leakage audit.
Software versions used: PyTDC 1.1.15, RDKit 2026.3.6, scikit-learn 1.6.1, XGBoost 3.4.1, NumPy 2.1.3,
pandas 2.2.3 (Python on Google Colab).

## Splits (splits/)
Random 5x5 CV, scaffold 5x5 CV (each acyclic molecule its own group), cluster 5x5 CV (Butina,
Tanimoto >= 0.4), plus fixed scaffold and random train/valid/test holdouts (209/19/46).
CV split files (JSON) hold row indices into data/curated_carcinogens.csv, which is the same row order
as splits/holdout_splits.csv.

## Main results (MCC, mean +/- sd over 5 repeats of pooled out-of-fold predictions)
MACCS + Random Forest   scaffold 0.640 +/- 0.021 | random 0.654 +/- 0.021 | cluster 0.602 +/- 0.041
RDKit desc + SVM        scaffold 0.624 +/- 0.031 | random 0.636 +/- 0.029 | cluster 0.557 +/- 0.056
Morgan + Random Forest  scaffold 0.622 +/- 0.015 | random 0.630 +/- 0.025 | cluster 0.553 +/- 0.006
GIN ensemble (2 seeds)  scaffold 0.503 +/- 0.022 | random 0.512 +/- 0.069 | not run
- A small, untuned GNN trailed the best classical model by about 0.14 MCC (all 5 repeats).
- Label-shuffling control: MCC about 0 (max +0.117) versus about 0.62 with real labels.
- Performance leans on two analogue series (aromatic azo dyes, alkyl hydrazines; 23 molecules, 22
  carcinogens). Without them AUROC is about 0.77-0.82.
- Error by similarity to training set: about 17% (low), 11% (mid), 5% (high).
- Structural alerts: hydrazine 8/8, azo 14/15, alkyl halide 4/5 carcinogens. The epoxide pattern
  matched 4 large natural-product-like molecules (all non-carcinogens); the narrowed Michael-acceptor
  pattern (post-hoc) matched 2 (both non-carcinogens).

## Reproduce
bash "model code/run_all.sh"
(on plain Windows, run the pip and python lines inside run_all.sh by hand)
Screening tool: python "model code/predict.py" --smiles "CCNN"
Full analysis notebook (threshold tuning, cluster-CV evaluation, series ablation, calibration,
applicability domain): model code/MINI_PROJECT.ipynb
Fresh-run check: 01_curate_and_split.py regenerated the random and scaffold CV splits identically.

## Checkpoints
checkpoints/final_* were trained on ALL 274 molecules, for the screening tool only. Report performance
only from results/ (cross-validation and holdout files), never from these models.

## Limitations
Small dataset (58 positives); reliance on two analogue series; shortcut learning on sulfonate/N-N
patterns; scaffold CV mildly optimistic versus cluster CV; fixed hyperparameters; GNN untuned;
structural-alert patterns unverified against the primary literature; no experimental validation.
Not for regulatory or clinical use.

## AI assistance
Code and analysis planning were assisted by Claude (Anthropic). Scientific decisions, interpretation
and conclusions are the team's.