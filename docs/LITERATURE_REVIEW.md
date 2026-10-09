# Literature review and assumptions

Challenge 59 (Safety), AI in Drug Discovery. Tathya Oza (BT26BTNCL11001). Educational and research use only.

## 1. Scope and method

This is a focused review, not a systematic one. It is based on about ten targeted web searches and on abstracts and summary pages of the papers cited, not on full texts. Statements about each paper describe what it reports in its abstract or summary. We make no claim that the combination of analyses in this project is new.

## 2. Why structure-based prediction, and structural alerts

Rodent carcinogenicity bioassays take years and use large numbers of animals, so a structure-based flag that is available before any synthesis or testing has clear value.

The oldest approach is the structural alert, a substructure associated with DNA-reactive chemistry. Ashby and Tennant (1988) examined 222 chemicals tested by the US National Cancer Institute and National Toxicology Program (NCI/NTP) and related chemical structure, Salmonella mutagenicity and rodent carcinogenicity. Later work encoded such knowledge in rule sets. The Benigni-Bossa rulebase, implemented in the Toxtree software, contains alerts for genotoxic and non-genotoxic carcinogenicity, and it sends aromatic amines and alpha,beta-unsaturated aldehydes on to QSAR models for a more refined assessment (Benigni & Bossa, 2008). Plošnik and colleagues (2016) review these alerts together with their mechanisms, chiefly the formation of adducts between reactive chemicals and DNA. A later review proposed a further set of structural alerts for non-genotoxic carcinogenicity (Benigni, Bossa & Tcheremenskaia, 2013). Benigni and colleagues (2007) surveyed the (Q)SAR models for mutagens and carcinogens that had accumulated by then.

Alerts describe DNA-reactive chemistry, so they are expected to miss carcinogens that act through other routes.

## 3. Early machine learning on rodent bioassay data

Learning from rodent data has a long history at the scale of a few hundred chemicals, similar to ours. The CASE method identified structural determinants linked to non-genotoxic carcinogens as well as universal and species-specific ones (Rosenkranz & Klopman, 1990), which shows that statistical learning can go beyond classical alerts. A knowledge-based rule-induction study carried out with toxicologists demonstrated the usefulness of rule induction for predicting rodent carcinogenicity (Lee et al., 1998). Bahler and Bristol (1999) built decision trees and rule sets from NTP bioassay results on 226 test agents and estimated error with ten-fold cross-validation.

## 4. Datasets and modern models

Our dataset is TDC `Carcinogens_Lagunin` (Huang et al., 2021). The TDC page cites Lagunin et al. (2009), who predicted rodent carcinogenicity with the PASS software and a second method (CISOC-PSCT), and Cheng et al. (2012, admetSAR).

The same group later released ROSC-Pred, a web service for species-, sex- and organ-specific rodent carcinogenicity. It uses PASS with multilevel neighbourhoods of atoms descriptors, was trained on 1,011 organic compounds from two-year rodent bioassays, and reports cross-validated accuracy of about 79% (leave-one-out) and 78% (ten-fold) (Lagunin et al., 2018). It addresses a different, organ-specific task on different data, so those figures cannot be compared with our MCC values.

DeepCarc uses deep-learning-based model-level representations on the NCTRlcdb database of 999 compounds, with carcinogenicity signals combined from rats and mice, and its authors report that it outperformed four advanced deep-learning QSAR models on the test set (Li et al., 2021).

## 5. Descriptor-based versus graph-based models

The GIN architecture used in this project is a graph network whose discriminative power equals that of the Weisfeiler-Lehman graph isomorphism test (Xu et al., 2019). A broad comparison on 11 public datasets found that descriptor-based models (SVM, XGBoost, random forest, DNN) outperformed graph-based models on average in accuracy and computational efficiency, although some graph models did well on larger or multi-task datasets (Jiang et al., 2021). Our result, with fingerprint models beating a small GIN on 274 molecules, is consistent with that picture.

## 6. Evaluation practice

Small datasets are often evaluated with a single split or plain random cross-validation. Wallach and Heifets (2018) introduced AVE bias to measure redundancy between training and validation data. They found that it correlates strongly with the performance of ligand-based methods across benchmarks, which suggests that reported performance can reflect overfitting to benchmarks. There is a counterpoint: a study of protein-ligand data found that debiasing typically reduced accuracy on distant held-out sets, and that performance after debiasing does not represent generalization (Sundar et al., 2020). No single split is therefore a perfect test, which is why this project used random, scaffold and cluster splits and reported each test molecule's similarity to the training set.

For the metric, Chicco and Jurman (2020) argue that MCC gives a high score only when all four confusion-matrix categories (true and false positives and negatives) are predicted well, in proportion to class sizes. This is why we used MCC as the primary metric on imbalanced data.

## 7. Where this project sits

In our targeted searches we did not find a carcinogenicity study that combined scaffold- and cluster-grouped validation, a classical-versus-GNN comparison on a small public set, and a structural-alert cross-check. The searches were limited, so this is not evidence that none exists, and we make no novelty claim. A systematic search (PubMed and Google Scholar, using terms such as "carcinogenicity prediction scaffold split", "small dataset carcinogenicity graph neural network" and "structural alerts machine learning comparison") would be needed before any such claim.

## 8. Problem-specific assumptions

1. TDC's labels are correct as provided, and each label is a rodent bioassay outcome, not a measure of human risk.
2. Non-stereo canonical SMILES carry enough information for this task.
3. A Tanimoto similarity of 0.4 on Morgan fingerprints is a reasonable boundary for "same chemical series" when grouping molecules for the cluster splits.
4. Our structural-alert patterns approximate the published alerts, but they have not been checked against the primary alert literature.
5. Fixed hyperparameters give a fair, if untuned, comparison between model families.
6. Data provenance: the TDC page lists 278 compounds, whereas our download had 280 rows. We describe the labels as "as provided by TDC" and do not claim to know where the underlying labels originated beyond the papers cited on that page.
7. The test sets are small, so cross-validated results are the primary evidence and the fixed holdouts are supplementary.

## 9. References

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