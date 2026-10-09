#!/usr/bin/env bash
# Run from the repository root.  A GPU is only needed for the GNN step.
set -e
grep -viE '^pytdc' requirements.txt | sed 's/+cu[0-9]*//' > requirements_clean.txt
pip install -r requirements_clean.txt
pip install PyTDC==1.1.15 --no-deps
pip install fuzzywuzzy huggingface_hub openpyxl tqdm requests
cd "model code"
python 01_curate_and_split.py
python 02_train_classical.py
python 03_train_gnn.py
python 04_alerts_and_sanity.py
python predict.py --smiles "CCNN" "c1ccccc1N"