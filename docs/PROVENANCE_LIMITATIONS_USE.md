# Data provenance, limitations, uncertainty and responsible use

# Data provenance, limitations, uncertainty and responsible use

## Data provenance
- Source: TDC `Carcinogens_Lagunin`, loaded with PyTDC 1.1.15 (`tdc.single_pred.Tox`).
  recorded in the project's environment file.
- The TDC page cites Lagunin et al. (2009) and Cheng et al. (2012, admetSAR) for this dataset.
- Size: the TDC page lists 278 compounds; our download contained 280 rows. We report 280 as received
  and do not know the cause of the difference.
- Origin of labels: we could not confirm where the underlying labels originally came from beyond those
  citations. Describe them as "labels as provided by TDC" until the 2009 paper has been read.
- Licence: the TDC page's licence line reads "Not Specified. CC BY 4.0". Recheck the licence before
  redistributing the data.
- TDC's own official split shares molecules between sets after standardization (1 between train and
  test, 3 between train and validation). Results on that split are therefore not directly comparable
  with ours, which were built after de-duplication.

## Limitations
1. Only 274 molecules and 58 positives; every estimate is noisy.
2. Much of the signal sits in two analogue series (23 molecules holding 22 of the 58 carcinogens).
3. The model shows shortcut learning: several of its most-used features are sulfonate and N-N keys,
   which act as proxies for the dye series.
4. Scaffold CV was mildly optimistic compared with cluster CV, and the cluster threshold (Tanimoto
   0.4) is gentle: 141 of 187 clusters are singletons.
5. The GNN was not evaluated under cluster splits.
6. The holdout sets (46 test molecules) are noisy.
7. Hyperparameters were fixed and the GNN was untuned, so the comparison says nothing about the best
   achievable performance of either model family.
8. The alert patterns are ours and unverified against the primary literature. The broad
   Michael-acceptor and epoxide patterns were poor, and the narrowed Michael-acceptor pattern is
   post-hoc.
9. Of the 26 carcinogens missed by the best model, 12 carry no alert at all, so mechanisms other than
   DNA reactivity are probably not captured. We did not check each compound.
10. The labels are rodent bioassay outcomes and say nothing about dose, metabolism or human risk.
11. The screening tool can give confident-looking scores to small molecules far from the training data
    (for example ethanol scored 0.668; we did not check this against a reference). Its reliability
    cutoffs (similarity 0.24 and 0.35) come from fold-specific training sets, so on the full training
    set they run slightly optimistic.
12. No experimental validation was performed.

## Uncertainty
Quantified through run-to-run standard deviations over 5 repeats, bootstrap intervals on holdouts,
the Brier score and reliability curve (0.092 against 0.167 for always predicting the base rate), error
by similarity group (17.2%, 11.4%, 4.8%), and selective prediction. All of it is exploratory, and none
of it is a formal coverage guarantee.

## Responsible use
- This work is for research and teaching only.
- It must not be used for regulatory submissions, clinical decisions, or to declare any chemical
  safe.
- A "non-carcinogen" call is not evidence of safety. A "carcinogen" call is a screening flag, not a
  finding.
- Any confirmation needs toxicological expertise and experiment.
- The similarity of a query molecule to the training data should be checked and reported with every
  prediction. For molecules unlike the training set the output is weak evidence.
- No animal experiments were performed in this study.

## AI assistance
Code, analysis planning and drafting were assisted by Claude (Anthropic). Scientific decisions,
interpretation and conclusions are the team's.
