# Manuscript material

## Title (draft)
Small-data carcinogenicity prediction under scaffold and cluster validation, with uncertainty analysis and a structural-alert cross-check

## Abstract
Rodent carcinogenicity bioassays are slow and costly, and public labelled data are small. We curated the TDC Carcinogens_Lagunin set (280 rows, 274 unique molecules, 21.2% carcinogens) and compared twelve classical feature-model combinations with a small graph neural network under repeated random, scaffold-grouped and cluster-grouped cross-validation, reporting pooled out-of-fold MCC, AUROC and balanced accuracy. A Random Forest on MACCS keys performed best (MCC 0.64 on scaffold and 0.60 on cluster cross-validation), and the GNN trailed by about 0.14 MCC. Label shuffling reduced MCC to about zero, confirming real signal. Performance depended heavily on two analogue series (sulfonated azo dyes and alkyl hydrazines; 23 molecules holding 22 of 58 carcinogens); without them AUROC fell to about 0.77 to 0.82. Error rose from about 5% to about 17% as similarity to the training set decreased. Structure-only prediction is informative here but dependable mainly near known chemistry.

## Contributions we can defend (and no more)
1. A leakage-audited benchmark, including the finding that TDC's official split shares molecules across sets (1 train/test, 3 train/valid after standardization).
2. A like-for-like comparison of twelve classical combinations and a GNN under random, scaffold and cluster validation.
3. A quantified dependence on two analogue series.
4. An applicability-domain and calibration analysis, and a small research screening tool.

## Before any submission
A novelty claim needs a real literature search, and a journal would want external validation, which was not done.
