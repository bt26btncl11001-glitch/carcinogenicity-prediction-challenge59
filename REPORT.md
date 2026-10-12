# Final project report
# Small-data carcinogenicity prediction under scaffold and cluster validation, with uncertainty analysis and a structural-alert cross-check

Challenge 59 (Safety), AI in Drug Discovery. Tathya Oza (BT26BTNCL11001). Educational and research use only.

## Abstract

Rodent carcinogenicity bioassays are slow and costly, and public labelled data are small. We curated the TDC Carcinogens_Lagunin set (280 rows, 274 unique molecules, 21.2% carcinogens) and compared twelve classical feature-model combinations with a small graph neural network under repeated random and scaffold-grouped cross-validation, with the three best classical models also evaluated under cluster-grouped cross-validation. We report pooled out-of-fold MCC, AUROC and balanced accuracy. A Random Forest on MACCS keys performed best (MCC 0.64 on scaffold and 0.60 on cluster cross-validation), and the GNN trailed by about 0.14 MCC. Label shuffling reduced MCC to about zero, confirming real signal. Performance depended heavily on two analogue series (sulfonated azo dyes and alkyl hydrazines; 23 molecules holding 22 of 58 carcinogens); without them AUROC fell to about 0.77 to 0.82. Error rose from about 5% to about 17% as similarity to the training set decreased. Structure-only prediction is informative here but dependable mainly near known chemistry.

## 1. Introduction and literature review

Rodent carcinogenicity bioassays are among the slowest and most expensive studies in drug and chemical safety assessment. A two-year bioassay uses large numbers of animals and takes years to complete. An imperfect early flag, available from structure alone before any synthesis or testing, is therefore worth having.

The oldest approach is the structural alert, a substructure known to be associated with DNA-reactive chemistry. Ashby and Tennant (1988) examined 222 chemicals tested by the US National Cancer Institute and National Toxicology Program and related chemical structure, Salmonella mutagenicity and rodent carcinogenicity. Later work turned this knowledge into rule sets. The Benigni-Bossa rulebase, implemented in the Toxtree software, encodes alerts for genotoxic and non-genotoxic carcinogenicity and sends aromatic amines and α,β-unsaturated aldehydes on to QSAR models for a more refined assessment (Benigni & Bossa, 2008). Plošnik and colleagues (2016) review these alerts together with the mechanisms behind them, chiefly the formation of adducts between reactive chemicals and DNA. Benigni and colleagues (2007) surveyed the (Q)SAR models for mutagens and carcinogens that had accumulated by then.

Alerts have a built-in limit. They encode DNA-reactive chemistry, so they are expected to miss carcinogens that act through other routes. Statistical models can in principle learn patterns beyond classical alerts. Rosenkranz and Klopman (1990) applied the CASE method to rodent carcinogen data and reported structural determinants associated with non-genotoxic carcinogens as well as universal and species-specific ones. The dataset used here comes from Lagunin and colleagues (2009), who predicted rodent carcinogenicity with the PASS software and a second method (CISOC-PSCT), and it is distributed through the Therapeutics Data Commons toxicity suite (Huang et al., 2021). The same group later released ROSC-Pred, a web service for species-, sex- and organ-specific carcinogenicity trained on 1,011 compounds, with reported cross-validated accuracy of about 78 to 79% (Lagunin et al., 2018). Those numbers come from leave-one-out and ten-fold cross-validation on a different, organ-specific task and cannot be compared with ours.

Two practical problems motivated this project. First, labelled carcinogenicity data are scarce, and small datasets are often evaluated with a single split or plain random cross-validation, which says little about performance on new chemistry. Benchmarking studies in cheminformatics have argued that random splits can reward memorizing near neighbours (Wallach & Heifets, 2018). Second, structural alerts and machine-learning models are seldom compared on the same data under the same validation.

We therefore set out to answer two questions. (1) How reliable is structure-only prediction on unseen chemistry? (2) With so little data, which approach generalizes better: classical models on fingerprints, or a graph neural network? To answer them we curated the dataset with a documented leakage audit and evaluated twelve classical feature-model combinations and a small graph neural network under repeated random and scaffold-grouped cross-validation, with the three best classical models also evaluated under cluster-grouped validation. We added a negative control, analysed calibration and applicability domain, cross-checked the model against a fixed list of structural alerts, and built a small screening tool for research use.

This is a focused review, not a systematic one. We make no claim that this combination of analyses is new, and any novelty statement would need a systematic literature search.

### 1.1 Related work

This subsection adds the machine-learning and evaluation literature to the structural-alert background above. It is based on about ten targeted web searches and on abstracts and summary pages, not full texts, and it is not a systematic review.

A later review proposed a further set of structural alerts for non-genotoxic carcinogenicity (Benigni, Bossa & Tcheremenskaia, 2013).

**Early machine learning on rodent bioassays.** Learning from rodent data has a long history at the scale of a few hundred chemicals, similar to ours. The CASE method identified structural determinants linked to non-genotoxic carcinogens as well as universal and species-specific ones (Rosenkranz & Klopman, 1990). A knowledge-based rule-induction study with toxicologists demonstrated the usefulness of rule induction for predicting rodent carcinogenicity (Lee et al., 1998). Bahler and Bristol (1999) built decision trees and rule sets from NTP bioassay results on 226 test agents and estimated error with ten-fold cross-validation.

**Modern models and services.** ROSC-Pred uses PASS with multilevel neighbourhoods of atoms descriptors, was trained on 1,011 organic compounds from two-year rodent bioassays, and reports cross-validated accuracy of about 78 to 79% (Lagunin et al., 2018). It predicts species-, sex- and organ-specific outcomes, which is a different task from ours. DeepCarc uses deep-learning-based model-level representations on the NCTRlcdb database of 999 compounds, with carcinogenicity signals combined from rats and mice, and its authors report that it outperformed four advanced deep-learning QSAR models on the test set (Li et al., 2021).

**Descriptor-based versus graph-based models.** The GIN architecture used here is a graph network whose discriminative power equals that of the Weisfeiler-Lehman test (Xu et al., 2019). A broad comparison on 11 public datasets found that descriptor-based models outperformed graph-based models on average in accuracy and efficiency, although some graph models did well on larger or multi-task datasets (Jiang et al., 2021). Our result, with fingerprint models beating a small GIN on 274 molecules, is consistent with that picture.

**Evaluation practice.** Wallach and Heifets (2018) introduced AVE bias to measure redundancy between training and validation data. They found that it correlates strongly with the performance of ligand-based methods across benchmarks, which suggests that reported performance can reflect overfitting to benchmarks. There is a counterpoint: a study of protein-ligand data found that debiasing typically reduced accuracy on distant held-out sets, and that performance after debiasing does not represent generalization (Sundar & Colwell, 2020). No single split is therefore a perfect test, which is why we used random, scaffold and cluster splits and reported similarity to the training set. For the metric, Chicco and Jurman (2020) argue that MCC gives a high score only when all four confusion-matrix categories are predicted well, in proportion to class sizes, which is why we chose MCC as the primary metric.

**Where this project sits.** In our targeted searches we did not find a carcinogenicity study that combined scaffold- and cluster-grouped validation, a classical-versus-GNN comparison on a small public set, and a structural-alert cross-check. The searches were limited, so this is not evidence that none exists. We make no novelty claim.

## 2. Data and splits

**Data.** We used TDC `Carcinogens_Lagunin`. The TDC page lists 278 compounds, and our download contained 280 rows; we report what we received. All SMILES parsed. Curation kept the largest fragment, neutralised charges, discarded molecules without carbon (none) and wrote canonical SMILES without stereochemistry. Two duplicate rows with consistent labels were collapsed, and two structures that appeared with conflicting labels were removed, leaving **274 molecules: 58 carcinogens and 216 non-carcinogens (21.2%)**.

TDC's own official split, after the same standardization, shares molecules between sets (1 between train and test, 3 between train and validation). Our splits were built after de-duplication and pass a leakage audit.

The dataset is structurally diverse. It contains 184 Bemis-Murcko scaffolds, 148 of them (80%) occurring once, and 19 molecules with no ring. Because all 19 would otherwise form one large group, each acyclic molecule was treated as its own group in the scaffold splits.

**Splits.** All splits were generated once, saved, and reused by every model.

| Split | Design |
|---|---|
| Random CV | 5-fold stratified, 5 repeats |
| Scaffold CV | 5-fold grouped by scaffold, class-stratified, 5 repeats |
| Cluster CV | 5-fold grouped by Butina clusters (Tanimoto >= 0.4, 187 clusters, 141 singletons, largest 10 molecules), 5 repeats |
| Scaffold holdout | train 209 (39 carcinogens), validation 19 (5), test 46 (14) |
| Random holdout | train 209 (44), validation 19 (4), test 46 (10) |

Audits confirmed no shared molecule in any split, and no shared scaffold or cluster (as we defined them) in the grouped splits. Scaffold-CV test folds contain 5 to 23 carcinogens (mean 11.6), so we pooled the five test folds of each repeat before computing metrics.

**Are the grouped splits harder?** Nearest-neighbour Tanimoto similarity from test molecules to their training set:

| Split | Mean | Median | Test molecules below 0.4 |
|---|---|---|---|
| Random CV | 0.405 | 0.353 | 54% |
| Scaffold CV | 0.330 | 0.286 | 73% |
| Random holdout | 0.407 | 0.353 | 59% |
| Scaffold holdout | 0.345 | 0.292 | 67% |

Even the random split has a median similarity of only 0.35, which anticipates why the scaffold-versus-random gap is small.

## 3. Methods

**Features.** Morgan count fingerprints (radius 2, 2048 bits); MACCS keys (166 keys; RDKit returns 167 bits with bit 0 unused), used unscaled; and RDKit 2D descriptors with missing or infinite values set to zero and a signed log transform. Scaling happened inside each model pipeline, so scalers were fitted on training folds only.

**Classical models.** Logistic regression, RBF-SVM, Random Forest (500 trees) and XGBoost (300 trees, depth 4), all class-balanced with fixed hyperparameters, so baselines are comparable and untuned.

**Graph neural network.** A GIN-style network with 3 layers, 64 hidden units, residual connections, dropout 0.2, AdamW, 100 epochs and a class-weighted loss, on 30 atom features (element, degree, charge, hybridization, hydrogen count, aromaticity, ring membership), with two seeds averaged per fold.

**Evaluation.** MCC is the primary metric, with AUROC, AUPRC and balanced accuracy alongside. Cutoffs were 0.5 for probabilities and 0 for the SVM margin. Metrics were computed on pooled out-of-fold predictions per repeat and summarised as mean and standard deviation over 5 repeats. Holdout results carry 95% bootstrap intervals (1,000 resamples) and are treated as supplementary. The GNN was evaluated on random and scaffold splits only.

**Controls and analyses.** Label shuffling; nested threshold tuning; removal of the two dominant analogue series from out-of-fold predictions; calibration (Brier score, reliability curve); applicability domain (similarity to the training set); and selective prediction. These are described in `METHODOLOGY.md`.

**Structural alerts.** Seventeen alert patterns were written before looking at results. We counted carcinogens among matching molecules and tested enrichment with one-sided Fisher exact tests. The patterns have not been checked against the primary alert literature.

**Screening tool.** `predict.py` standardizes SMILES and scores them with the final MACCS-keys Random Forest trained on all 274 molecules. For each molecule it reports the probability, a call (cutoff 0.5), the nearest-neighbour similarity to the training set, and a reliability label ("higher" at similarity 0.35 or above, "medium" from 0.24, "lower" below that).

## 4. Results

### 4.1 Cross-validated performance

**Table 1. Pooled out-of-fold performance, mean over 5 repeats (MCC / AUROC).**

| Features + model | Scaffold CV | Random CV |
|---|---|---|
| MACCS + Random Forest | 0.640 / 0.854 | 0.654 / 0.850 |
| RDKit descriptors + SVM | 0.624 / 0.882 | 0.636 / 0.887 |
| Morgan + Random Forest | 0.622 / 0.874 | 0.630 / 0.885 |
| RDKit descriptors + Random Forest | 0.605 / 0.855 | 0.624 / 0.867 |
| Morgan + Logistic regression | 0.571 / 0.841 | 0.564 / 0.847 |
| Morgan + XGBoost | 0.562 / 0.873 | 0.571 / 0.859 |
| MACCS + SVM | 0.555 / 0.849 | 0.593 / 0.842 |
| MACCS + XGBoost | 0.551 / 0.826 | 0.566 / 0.815 |
| RDKit descriptors + XGBoost | 0.544 / 0.833 | 0.595 / 0.848 |
| MACCS + Logistic regression | 0.521 / 0.837 | 0.542 / 0.827 |
| RDKit descriptors + Logistic regression | 0.513 / 0.852 | 0.551 / 0.855 |
| Morgan + SVM | 0.417 / 0.740 | 0.522 / 0.821 |
| **GIN ensemble** | **0.503 / 0.828** | **0.512 / 0.835** |

Run-to-run standard deviations of MCC were about 0.02 to 0.03 for the top models (MACCS + RF 0.021 and 0.021; RDKit descriptors + SVM 0.031 and 0.029; Morgan + RF 0.015 and 0.025), so the top three are not clearly separated. Random Forest ranked first or second within every feature set. Morgan + SVM was clearly worst.

**Cluster CV (three best classical models).** MACCS + RF 0.602 ± 0.041 (AUROC 0.827); Morgan + RF 0.553 ± 0.006 (0.864); RDKit descriptors + SVM 0.557 ± 0.056 (0.868). The GNN was not run on these splits.

**GNN versus the best classical model.** The paired MCC difference was −0.137 (scaffold) and −0.141 (random), negative in all five repeats on both. With the best possible cutoff, chosen on the test predictions themselves (an optimistic ceiling), the GNN reached 0.563 and 0.592 (best cutoffs about 0.93 and 0.83), still below the classical models.

### 4.2 Fixed holdouts (supplementary, 46 test molecules)

On the scaffold holdout, the best-scoring model was Morgan + XGBoost (MCC 0.795, 95% interval 0.58 to 0.96), followed by Morgan + RF (0.791) and RDKit descriptors + SVM (0.749); Morgan + SVM was at chance (0.091, AUROC 0.50). On the random holdout the best were RDKit descriptors + SVM and MACCS + SVM (0.735 each). The GNN reached 0.749 (interval 0.51 to 0.94) on the scaffold holdout and 0.586 (0.27 to 0.86) on the random holdout. These rankings differ from cross-validation, as expected for 46 molecules. Models were selected from cross-validation, never from these holdouts.

### 4.3 Is the signal real?

With labels shuffled, mean MCC for the three best models (scaffold CV) was −0.019 (MACCS + RF), −0.062 (RDKit descriptors + SVM) and −0.027 (Morgan + RF); the maximum over 45 shuffled runs was +0.117, against about 0.62 with real labels. Choosing the cutoff inside each training fold did not help the classical models (MCC change −0.01 to −0.03 for four models), so default cutoffs were kept.

### 4.4 Dependence on two analogue series

The azo and hydrazine molecules (23 molecules, 22 of the 58 carcinogens) carry a large share of the performance. Removing them from the existing out-of-fold predictions (no retraining):

| Model | MCC all → without | AUROC all → without |
|---|---|---|
| MACCS + RF | 0.640 → 0.444 | 0.854 → 0.773 |
| Morgan + RF | 0.622 → 0.423 | 0.874 → 0.805 |
| RDKit descriptors + SVM | 0.624 → 0.447 | 0.882 → 0.818 |

Only 14% of the remaining 251 molecules are carcinogens, so part of the MCC drop is a prevalence effect, and AUROC is the fairer summary. The models were trained with the series included and only 36 carcinogens remain, so these figures are indicative.

### 4.5 Error analysis (MACCS + RF, scaffold CV, scores averaged over repeats)

True positives 32, false negatives 26, false positives 4, true negatives 212. Of the 20 highest-scoring molecules, 19 are carcinogens and all 20 carry an alert (15 aromatic azo dyes, 5 alkyl hydrazines). Of the 26 missed carcinogens, 12 carry no alert at all (for example an ethynyl-steroid and several nucleoside and drug-like molecules). One miss is 2-chloromethylpyridine, a benzylic halide, scored 0.10, probably because the dataset holds only one such example. The four false positives are two sulfonamide drugs resembling the dyes, one azo dye labelled non-carcinogen, and one small guanidine-type molecule. Several of the most-used MACCS keys encode sulfur-oxygen groups and N-N bonds. Sulfonates are not carcinogenic by themselves, so here they act as a proxy for "sulfonated azo dye", which is shortcut learning.

### 4.6 Calibration, applicability domain and selective prediction

The Brier score was 0.092 against 0.167 for always predicting the base rate. The reliability curve is decent (predicted 0.77 observed 0.79) with one overconfident middle bin (predicted 0.26, observed 0.14). The five repeats are not independent.

| Max similarity to training set | Error rate | MCC | AUROC |
|---|---|---|---|
| Low (0.09 to 0.24) | 17.2% | 0.47 | 0.83 |
| Mid (0.24 to 0.34) | 11.4% | 0.45 | 0.65 |
| High (0.35 to 0.81) | 4.8% | 0.87 | 0.97 |

The pattern is not perfectly monotone, and carcinogen rates differ between groups, so we treat it as a rule of thumb. Answering only when the predicted probability was below 0.25 or above 0.75 covered 85.5% of molecules and raised MCC from 0.640 to 0.717. This analysis is exploratory, used a fixed grid, and is not conformal prediction.

### 4.7 Structural alerts

The base carcinogen rate is 21.2%.

| Alert | n | Carcinogens | Rate | One-sided p |
|---|---|---|---|---|
| Hydrazine | 8 | 8 | 1.00 | < 0.0001 |
| Aromatic azo | 15 | 14 | 0.93 | < 0.0001 |
| Alkyl halide | 5 | 4 | 0.80 | 0.0078 |
| N-nitroso | 2 | 1 | 0.50 | 0.38 |
| Aromatic amine | 32 | 12 | 0.38 | 0.0185 |
| Aromatic nitro | 4 | 1 | 0.25 | 0.62 |
| Aldehyde | 4 | 1 | 0.25 | 0.62 |
| Michael acceptor (broad) | 35 | 3 | 0.09 | 0.99 |
| Epoxide / aziridine | 4 | 0 | 0.00 | 1.0 |
| Benzylic/allylic halide, alkyl sulfonate, PAH | 1 each | 1, 1, 0 | n/a | too few molecules |

Five of the seventeen alerts matched no molecule. With 12 alerts tested, the Bonferroni threshold is about p < 0.004, so only hydrazine and azo clearly survive; alkyl halide rests on five molecules and aromatic amine does not reach the corrected threshold. Rows with four or fewer molecules support no conclusion.

The broad Michael-acceptor pattern matches amides, esters and ring enones, so it is not a faithful alert. The four "epoxide" matches are large natural-product-like molecules, all non-carcinogens (judging from the SMILES, a terpenoid lactone, a macrolide, a tropane N-oxide and a germacranolide; identities not confirmed), so the pattern's result says nothing against the textbook epoxide alert. A narrowed Michael-acceptor pattern, written after seeing the data and therefore post-hoc, matched two molecules (both non-carcinogens); a second variant matched none.

The rule "carcinogen if any alert is present" gave MCC 0.303, AUROC 0.692 and balanced accuracy 0.678 on all 274 molecules (36% carry an alert; sensitivity 0.64, specificity 0.72). It involves no training. The broad Michael-acceptor pattern drags it down, so it is a pessimistic baseline.

### 4.8 Screening tool test

`CCNN` scored 0.990 (carcinogen) and was flagged as a training-set molecule; 4-aminoazobenzene scored 0.774 (carcinogen, medium reliability); paracetamol scored 0.216 (non-carcinogen); an invalid string returned "invalid input". Ethanol scored 0.668 with medium reliability (nearest similarity 0.250, just above the "lower" cutoff); we did not check this call against a reference. It shows that small molecules far from the training data can receive confident-looking scores.

## 5. Discussion

**Which approach generalizes better with limited data?** Fingerprint-based Random Forests and SVMs outperformed a small GIN ensemble under both random and scaffold cross-validation. The result is consistent with the expectation that learned representations need more data than 274 molecules provide, since a network that must discover its own atom features has little to learn from. Part of the GNN's MCC deficit came from its probability cutoff, and the gap in AUROC (0.83 against 0.85 for the best-MCC model and 0.88 for the best-AUROC model) was smaller than the gap in MCC, but the best-case cutoff still did not close the gap. The claim must stay narrow: this small, untuned GNN loses at this data size. We did not test a pretrained graph model fine-tuned on this data, a larger hyperparameter search, or richer bond features. The one clear loser among classical models was an RBF-SVM on standardized Morgan counts, a mismatch between kernel and a 2,048-dimensional sparse input.

**How reliable is structure-only prediction on unseen chemistry?** Moderately, and less than the headline numbers suggest.

*The signal is real.* Label shuffling brought MCC to about zero. The test detects label information leaking into features or training; it does not look for near-duplicate molecules across splits, which the leakage audit and similarity analysis address.

*The grouped splits are harder, but the scaffold-random gap is small.* The best model scored 0.640 on scaffold, 0.654 on random and 0.602 ± 0.041 on cluster-grouped validation. Clustering at a Tanimoto threshold keeps close analogues of a cluster's centre together but does not guarantee that no pair of similar molecules straddles a split. Only the Morgan Random Forest drop from scaffold to cluster validation clearly exceeds its run-to-run spread. The small scaffold-random gap fits the data: 80% of the scaffolds occur once, and even random splits have a median nearest-neighbour similarity of only 0.35, so random splits are already mostly unseen chemistry. One hypothesis is that the useful signal sits in functional groups that travel across scaffolds; we did not test that directly. A broader point follows from the cluster result: scaffold grouping alone can under-estimate distribution shift in sparse chemical spaces, because molecules with different scaffolds can still be close analogues.

*A large share of the performance comes from two chemical series.* All 8 hydrazines and 14 of 15 azo compounds are carcinogens, and these series dominate the top-scoring molecules. Without them AUROC falls to about 0.77 to 0.82. In this dataset the model's reliance on sulfonate and N-N features reflects shortcut learning.

*Reliability depends on similarity to the training set.* Error ranged from 17.2% for the least similar third of molecules to 4.8% for the most similar third, and calibration was reasonable. Selective prediction raised MCC to 0.717 at 85.5% coverage.

**Machine learning versus structural alerts.** Hydrazine and aromatic azo are strongly enriched, and the machine-learning models mainly rediscovered them while adding modest information beyond them. The simple alert rule (MCC 0.303) was far below the models, although our broad Michael-acceptor pattern makes it a pessimistic baseline. Twelve of the 26 missed carcinogens carry no alert, which fits the expectation that alerts and alert-driven models do not cover non-DNA-reactive mechanisms; we did not check the mechanism of each compound. A hybrid that adds alert features to the fingerprint model would be a natural next step, and it might have caught 2-chloromethylpyridine.

**A caution about single splits.** On the 46-molecule scaffold holdout the best-scoring classical model on that holdout (Morgan + XGBoost) reached MCC 0.795 (interval 0.58 to 0.96), the ranking differed from cross-validation, and the GNN scored 0.749 on the scaffold holdout but 0.586 on the random one. Cross-validation is the primary evidence here.

## 6. Limitations

1. Very little data: 274 molecules and 58 positives. Every estimate is noisy, and folds can contain as few as 5 carcinogens.
2. Heavy dependence on two analogue series (23 molecules holding 22 of 58 carcinogens); we cannot say how the model behaves on carcinogen classes the dataset barely contains.
3. Shortcut learning on sulfonate and N-N patterns.
4. Scaffold cross-validation was mildly optimistic compared with cluster cross-validation, our cluster threshold (Tanimoto 0.4) is gentle (141 of 187 clusters are singletons), and the GNN was not evaluated under cluster splits. The fixed holdouts are small and noisy.
5. Fixed hyperparameters and an untuned GNN: a fair like-for-like comparison, but not a statement about the best achievable performance of either family.
6. The structural-alert patterns are ours and unverified against the primary literature; several were too broad, and the narrowed pattern is post-hoc.
7. Data provenance: the TDC page lists 278 compounds against our 280 rows; the origin of the underlying labels beyond the cited papers is unconfirmed; TDC's official split shares molecules across sets.
8. The labels are rodent bioassay outcomes and say nothing about dose, metabolism or human relevance; a "non-carcinogen" label does not mean a chemical is safe.
9. The screening tool can give confident-looking scores to small molecules far from the training data, and its reliability cutoffs, taken from fold-specific training sets, run slightly optimistic on the full training set.
10. No experimental validation. The tool is for research and teaching only and must not be used for regulatory or clinical decisions.

## 7. Conclusion

On a small public carcinogenicity dataset, simple fingerprint models are the right-sized choice. A Random Forest on MACCS keys reached a pooled MCC of about 0.64 on scaffold-grouped and 0.60 on cluster-grouped cross-validation, it beat a small graph neural network by about 0.14 MCC, and its output passed a label-shuffling negative control. The reported performance reflects real signal, but it should be read with its context. A large share comes from two well-known chemical classes (sulfonated azo dyes and alkyl hydrazines), where the model largely rediscovers established structural alerts. Outside them, discrimination was weaker, and error grew as molecules became less similar to the training set.

The practical message is twofold. The model is a reasonable early flag for chemistry that resembles its training data, and the similarity of a query molecule to that data should be checked and reported with every prediction. For genuinely new chemistry, the output is weak evidence and needs toxicological judgment and experiment.

Natural next steps are an alert-augmented fingerprint model, fine-tuning a pretrained graph model on this data, stricter cluster thresholds, formal conformal prediction, a systematic literature search to place the work properly, and validation on an external carcinogenicity set.

## References

1. Lagunin A, et al. Computer-aided prediction of rodent carcinogenicity by PASS and CISOC-PSCT. QSAR & Combinatorial Science 28(8): 806-810 (2009).
2. Cheng F, et al. admetSAR: a comprehensive source and free tool for assessment of chemical ADMET properties. J Chem Inf Model (2012), 3099-3105.
3. Huang K, et al. Therapeutics Data Commons. arXiv:2102.09548 (NeurIPS 2021).
4. Ashby J, Tennant RW. Chemical structure, Salmonella mutagenicity and extent of carcinogenicity as indicators of genotoxic carcinogenesis among 222 chemicals tested in rodents by the U.S. NCI/NTP. Mutation Research 204(1): 17-115 (1988).
5. Benigni R, Bossa C. The Benigni/Bossa Rulebase for Mutagenicity and Carcinogenicity - A Module of Toxtree. JRC report JRC43157 (2008).
6. Plošnik A, Vračko M, Sollner Dolenc M. Mutagenic and carcinogenic structural alerts and their mechanisms of action. Arh Hig Rada Toksikol 67: 169-182 (2016).
7. Benigni R, Netzeva TI, Benfenati E, et al. The expanding role of predictive toxicology: an update on the (Q)SAR models for mutagens and carcinogens. J Environ Sci Health C 25(1): 53-97 (2007).
8. Lagunin A, Rudik A, Filimonov D, Druzhilovsky D, Poroikov V. ROSC-Pred: web-service for rodent organ-specific carcinogenicity prediction. Bioinformatics 34(4): 710-712 (2018).
9. Rosenkranz HS, Klopman G. New structural concepts for predicting carcinogenicity in rodents: an artificial intelligence approach. Teratogenesis, Carcinogenesis, and Mutagenesis 10(2): 73-88 (1990).
10. Wallach I, Heifets A. Most ligand-based classification benchmarks reward memorization rather than generalization. J Chem Inf Model 58(5): 916-932 (2018). doi:10.1021/acs.jcim.7b00403.
11. Lee Y, Buchanan BG, Aronis JM. Knowledge-based learning in exploratory science: learning rules to predict rodent carcinogenicity. Machine Learning 30 (1998).
12. Bahler D, Bristol DW. Prediction of chemical carcinogenicity in rodents by machine learning of decision trees and rule sets. AAAI Spring Symposium, SS-99-01 (1999).
13. Li T, Tong W, Roberts R, Liu Z, Thakkar S. DeepCarc: deep learning-powered carcinogenicity prediction using model-level representation. Front Artif Intell (2021).
14. Jiang D, Wu Z, et al. Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models. J Cheminform 13: 12 (2021).
15. Xu K, Hu W, Leskovec J, Jegelka S. How powerful are graph neural networks? ICLR 2019.
16. Sundar V, Colwell L. The effect of debiasing protein-ligand binding data on generalization. J Chem Inf Model 60: 56-62 (2020).
17. Chicco D, Jurman G. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics 21: 6 (2020).
18. Benigni R, Bossa C, Tcheremenskaia O. Nongenotoxic carcinogenicity of chemicals: mechanisms of action and early recognition through a new set of structural alerts. Chem Rev 113(5): 2940-2957 (2013).

## AI assistance

Code, analysis planning and drafting were assisted by Claude (Anthropic). Scientific decisions, interpretation and conclusions are the author's.
