# Technical and methodology documentation

# Technical and methodology documentation

## 1. Pipeline in one pass
Load TDC data → standardize and de-duplicate → compute scaffolds and clusters → generate and audit
splits → compute three feature sets → train 12 classical combinations and a GNN on identical splits →
pool out-of-fold predictions per repeat → summarise MCC, AUROC, AUPRC and balanced accuracy → run
controls (label shuffling, series ablation, threshold tuning) → analyse structural alerts,
calibration, applicability domain and selective prediction.

## 2. Data curation
Source: TDC `Carcinogens_Lagunin`, loaded with PyTDC. Steps, in order:
1. Parse each SMILES with RDKit (all 280 rows parsed).
2. Keep the largest fragment (removes salts).
3. Neutralise charges.
4. Drop molecules containing no carbon (none were dropped).
5. Write canonical SMILES without stereochemistry, so stereoisomers collapse into one entry.
6. Collapse duplicate rows that have consistent labels (2 rows removed).
7. Remove structures that appear with conflicting labels (2 structures removed).
Result: 274 molecules, 58 carcinogens and 216 non-carcinogens (21.2%).

## 3. Split generation
One script, fixed seeds (42, plus the repeat number for repeated splits).
- Random CV: 5-fold stratified, 5 repeats.
- Scaffold CV: 5-fold, grouped by Bemis-Murcko scaffold, class-stratified, 5 repeats. Each acyclic
  molecule is its own group, so the 19 ring-free molecules spread across folds instead of landing in
  one fold.
- Cluster CV: 5-fold, grouped by Butina clusters (Morgan fingerprints, Tanimoto similarity >= 0.4),
  5 repeats. 187 clusters, 141 of them singletons, largest cluster 10 molecules.
- Fixed holdouts (scaffold-grouped and random), sizes 209 / 19 / 46 (train / valid / test).
Splits are stored in `splits/` as JSON index lists (CV) and CSV files (holdouts).

## 4. Features
- Morgan count fingerprints: radius 2, 2048 bits, counts rather than presence only.
- MACCS keys: 166 structural keys (RDKit returns 167 bits; bit 0 is always empty), used unscaled.
- RDKit 2D descriptors: the full RDKit descriptor list; NaN or infinite values set to 0; signed log
  transform sign(x) * log(1 + |x|). Scaling happens inside each model pipeline, so scalers are fitted
  on training folds only.
- GNN atom features (30): element (C, N, O, S, P, F, Cl, Br, I, other), degree, formal charge,
  hybridization, hydrogen count, aromaticity, ring membership.

## 5. Models (fixed hyperparameters, no tuning)
- Logistic regression (C = 0.1, class-balanced), RBF-SVM (C = 1, class-balanced), Random Forest
  (500 trees, balanced subsample), XGBoost (300 trees, depth 4, learning rate 0.05, positive-class
  weight from the training fold).
- GNN: GIN-style network, 3 layers, 64 hidden units, residual connections, dropout 0.2, mean plus sum
  readout, AdamW (learning rate 0.002, weight decay 0.001), 100 epochs, batch size 32, class-weighted
  loss, 2 seeds per fold averaged.
- Cutoffs: 0.5 for probabilities, 0 for the SVM margin.

## 6. Evaluation
- Primary metric MCC (small, imbalanced data); also AUROC, AUPRC and balanced accuracy.
- Folds can contain as few as 5 carcinogens, so the five test folds of each repeat are pooled and
  metrics are computed on the pooled out-of-fold predictions. Results are mean and standard
  deviation over the 5 repeats.
- Holdout metrics carry 95% bootstrap intervals (1,000 resamples). Holdouts are treated as
  supplementary because they contain only 46 test molecules.
- The GNN was evaluated on the random and scaffold splits only. Cluster CV covers the three best
  classical models.

## 7. Leakage controls
1. Standardize and de-duplicate before splitting.
2. Assert that no canonical SMILES appears in both train and test, for every split.
3. Check that no scaffold or cluster (as defined by our scaffold and clustering procedures) is shared
   between train and test in the grouped splits. Clustering at a similarity threshold does not
   guarantee that no pair of similar molecules straddles a split.
4. Fit scalers only inside training folds.
5. Select models by cross-validation, never by the holdouts.
6. Tune thresholds only inside training folds. The one place where test predictions were used to pick
   a cutoff (the GNN ceiling analysis) is labelled an optimistic upper bound.
7. Final models trained on all 274 molecules are for the screening tool only. No performance claim
   may come from them.

## 8. Controls and robustness analyses
- Label shuffling (scaffold CV, three best models, three permutations): confirms that performance
  disappears when labels carry no information. It does not detect near-duplicate molecules across
  splits; that is covered by the leakage audit and the similarity analysis.
- Nested threshold tuning (inside training folds) for four classical models, three repeats.
- Series ablation: out-of-fold predictions recomputed without the azo and hydrazine molecules, no
  retraining.
- Cluster-grouped CV, to test whether scaffold grouping under-estimates distribution shift.
- Nearest-neighbour Tanimoto similarity from each test molecule to its training set.

## 9. Structural-alert analysis
17 alert patterns were written before looking at results, from general toxicology knowledge. For each,
we count carcinogens among matching molecules and test enrichment with a one-sided Fisher exact test.
With 12 alerts matching at least one molecule, the Bonferroni threshold is about p < 0.004. The alert
SMARTS have not been checked against Ashby and Tennant (1988) or the Benigni-Bossa rulebase and must
be before any citation. Narrowed Michael-acceptor patterns were written after seeing the data and are
labelled post-hoc.

## 10. Uncertainty and applicability domain
- Calibration: Brier score and reliability curve on pooled out-of-fold predictions (MACCS + Random
  Forest, scaffold CV, 5 repeats pooled; the repeats are not independent).
- Applicability domain: maximum Tanimoto similarity (Morgan, radius 2) of each test molecule to its
  own fold's training set, split into three equal-size groups.
- Selective prediction: answer only when the predicted probability is below 0.25 or above 0.75
  (confidence thresholds on a fixed grid; exploratory, not conformal prediction).

## 11. Assumptions
1. TDC labels are correct as provided and are rodent bioassay outcomes, not human risk.
2. Non-stereo canonical SMILES carry enough information.
3. Tanimoto similarity 0.4 on Morgan fingerprints is a reasonable "same chemical series" boundary.
4. Our alert patterns approximate published alerts.
5. Fixed hyperparameters give a fair, if untuned, comparison between model families.

## 12. Reproducibility
- `requirements.txt` holds pinned versions; PyTDC is installed separately with `pip install PyTDC
  --no-deps` because its full dependency set failed to build on Colab.
- `model code/run_all.sh` runs the pipeline scripts in order.
- The scripts were refactored from the notebook cells that produced every reported number, using the
  same seeds, but have not yet been run as scripts. The notebook is the verified source.
- The threshold-tuning, cluster-CV evaluation, series-ablation, calibration and applicability-domain
  analyses exist only in the notebook.
- Reproducibility check: run `01_curate_and_split.py` in a clean session and compare the new
  `splits/cv_scaffold_5x5.json` with the one from the notebook run.
