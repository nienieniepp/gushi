# gpt2-chinese-poem-generator

A reproducible deep learning project for **Chinese classical poetry generation** using **GPT-2**, **PyTorch**, and **Hugging Face Transformers**.

This repository is designed for a university AI course project with a clean, modular, and academic structure.

## 1. Project Goals

- Build a full pipeline from raw data to trained model and interactive demo.
- Fine-tune a GPT-2 style language model on Chinese classical poems from the `chinese-poetry` dataset.
- Provide transparent experiments via notebooks and script-based training for reproducibility.

## 2. Project Structure

```text
gpt2-chinese-poem-generator/
│
├── data/
│   ├── raw/                     # Original downloaded chinese-poetry files
│   └── processed/               # Cleaned training text and metadata
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb   # EDA + cleaning walkthrough
│   └── 02_train_gpt2.ipynb      # Training walkthrough
│
├── app/
│   ├── app.py                   # Flask demo app
│   └── templates/
│       └── index.html           # Web UI for poem generation
│
├── models/                      # Saved model/checkpoints/tokenizer
├── report/                      # Final report, figures, and tables
│
├── src/
│   ├── prepare_data.py          # Deterministic data cleaning script
│   ├── train.py                 # Fine-tuning script with HF Trainer
│   └── generate.py              # CLI generation script
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 3. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Data Preparation

1. Download `chinese-poetry` and place JSON files under `data/raw/`.
2. Run deterministic preprocessing:

```bash
python src/prepare_data.py \
  --input_dir data/raw \
  --output_file data/processed/poems.txt \
  --min_len 8 \
  --max_len 160
```

Output:
- `data/processed/poems.txt`: one cleaned poem per line.
- `data/processed/stats.json`: corpus statistics for reporting.

## 5. Model Training

Example fine-tuning command:

```bash
python src/train.py \
  --train_file data/processed/poems.txt \
  --model_name_or_path uer/gpt2-chinese-cluecorpussmall \
  --output_dir models/gpt2-poem \
  --block_size 128 \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --seed 42
```

Training artifacts are saved to `models/gpt2-poem/`.

## 6. Poem Generation

### CLI

```bash
python src/generate.py \
  --model_dir models/gpt2-poem \
  --prompt 春江花月夜 \
  --max_new_tokens 64
```

### Web App

```bash
cd app
python app.py
```

Open `http://127.0.0.1:5000` and generate poems from prompts.

## 7. Reproducibility Checklist

- Fixed random seed support in scripts.
- Explicit hyperparameter arguments.
- Deterministic preprocessing outputs.
- Clear separation of raw/processed/model artifacts.
- Notebook + script dual workflow for education and production.

## 8. Suggested Course Deliverables

Place final outputs under `report/`:
- `report.pdf` (method, experiments, error analysis)
- Figures: loss curves, sample generations
- Tables: hyperparameters and evaluation metrics

## 9. License and Data Note

Please verify and follow the license/usage policy of the upstream `chinese-poetry` dataset repository before distribution.
