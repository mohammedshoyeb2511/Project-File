# SNOMED CT + MIMIC-IV Graph Construction and R-GCN Model Training

This repository provides the full pipeline for constructing a unified clinical knowledge graph from **SNOMED CT** and **MIMIC-IV**, generating the required `graph.pt` file, and training an **R-GCN (Relational Graph Convolutional Network)** model.

The final model only requires **two files** to run:
- `model_components.py`
- `train_rcgn.py`

But before training, you **must generate the graph.pt file** using the preprocessing scripts.

---

## 1. Requirements Installation

Install all dependencies: requirements.txt

2. Input Data Needed
You must download the following manually (not included due to licensing/size):

I. SNOMED CT (U.S. Edition)
1. Relationship snapshot file
2. Description snapshot file

II. MIMIC-IV v3.1
 hosp/diagnoses_icd.csv.gz

3. Pipeline Execution Order

Run the scripts in this exact order to generate all required datasets and the final graph.pt file used for training.
Step 1 — Preprocess SNOMED CT relationships
Script: Snomed CT Dataset loading.py
This script loads SNOMED CT descriptions + relationships and produces a clean ontology dataset with all 213 relation types preserved.
Output: snomed_relations_full.tsv

Step 2 — Map MIMIC ICD-10 codes to SNOMED CT
Script: merge_mimic_to_icd_dataset.py
Maps ICD-10 codes to SNOMED CT concepts and extracts co-occurrence relations.
Output: mimic_snomed_pairs.tsv

Step 3 — Merge SNOMED and MIMIC relations
Script: merge_snomed_and_mimic.py
Combines ontology edges + clinical co-occurrence edges into one dataset.
Output: merged_relations.tsv

Step 4 — Clean conflicts and duplicates
Script: conflits.py
Removes duplicate edges, symmetric duplicates, and rows with missing terms.
Output: cleaned_relations.tsv

Step 5 — Create graph triples and generate graph.pt
Script: Dataset_creation.py
(This is the MOST IMPORTANT preprocessing script.)
It converts the merged dataset into machine-readable triples, creates ID maps, and generates the PyTorch graph file.
Outputs:

graph.pt             <-- REQUIRED for model training
node_id_map.csv
rel_id_map.csv
triples_final.csv

Once this step is finished, you now have everything needed to train the model.

4. Running the R-GCN Model
**After generating graph.pt**, only two files are required:
Required Files:
graph.pt
node_id_map.csv
rel_id_map.csv
model_components.py
train_rcgn.py

Step 6 — Train the R-GCN Model
Script: train_rcgn.py
This script:
Loads graph.pt
Builds the model defined in model_components.py
Trains using negative sampling
Computes MRR, Hits@1, Hits@10
Saves the final trained model
Run:
python train_rcgn.py

Training Outputs:
pure_rgcn_final.pt        # Trained R-GCN model
history_loss.json         # Loss curve logs
history_metrics.json      # MRR / Hits@K logs
