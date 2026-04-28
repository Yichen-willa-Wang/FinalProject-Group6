# Project 1: Encoder Application — Social Media Shortage Detection

## 1. Project Overview

This project implements an encoder-based NLP application to detect drug shortage-related discussions from Reddit data.

The goal is to transform large-scale unstructured social media text into structured shortage signals using a transformer-based classification model.

This is a standalone encoder application and does not depend on the decoder project.

---

## 2. Business Problem

Drug shortages affect:

- patient access to medication
- treatment continuity
- switching behavior
- healthcare decision-making

Social media (Reddit) contains early signals such as:

- “cannot find medication”
- “pharmacy out of stock”
- “switching drugs”
- “delay in refill”

However, these signals are buried in noisy text.

This project builds an NLP classifier to automatically detect shortage-related posts.

---

## 3. Task Definition

Binary classification:

- `1` → shortage-related
- `0` → non-shortage-related

---

## 4. Data

### 4.1 Labeled Data

Two manually labeled datasets:

- `1000samples.xlsx`
- `df_nokey_results.xlsx`

Columns used:

| column | description |
|------|------------|
| text | Reddit text |
| label | shortage label |
| reason | annotation explanation |

### 4.2 Full Dataset (for prediction)

- `df_comments.csv`
- `df_submissions.csv`

---

## 5. Full Pipeline (IMPORTANT)

This script is **NOT just a model training script**.  
It is a **full pipeline**, including:

1. Data loading
2. Data cleaning
3. Data augmentation (word swap)
4. Model training (baseline)
5. Model training (improved weighted model)
6. Model saving
7. Full corpus prediction
8. Topic modeling (LDA)
9. Time-series analysis
10. Drug-specific analysis

---

## 6. Main Script

```
encoder_application/shortage_identification.py
```

Run everything in ONE script.

---

## 7. Step-by-Step Code Logic

### Step 1 — Load Data

```python
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)
```

### Step 2 — Label Cleaning

```python
data['label'] = data['label'].map({'YES': 1.0, 'NO': 0.0})
```

### Step 3 — Data Augmentation (Important)

Function:

```python
swap_words_in_corpus()
```

Purpose:

- randomly replace words like "shortage"
- reduce keyword leakage
- improve generalization

---

### Step 4 — Train/Test Split

```python
60% train
20% validation
20% test
```

---

### Step 5 — Tokenization

```python
AutoTokenizer.from_pretrained("roberta-base")
```

Two versions:

- max_length = 128
- max_length = 512

---

### Step 6 — Model Training (Baseline)

```python
Trainer(...)
```

- CrossEntropyLoss
- No class weights

---

### Step 7 — Model Training (Final Version)

Key improvements:

#### (1) Class Imbalance Handling

```python
compute_class_weight()
```

#### (2) Custom Loss

```python
class WeightedTrainer(Trainer)
```

#### (3) Freeze Layers

```python
freeze first 8 transformer layers
```

#### (4) Early Stopping

```python
EarlyStoppingCallback(patience=3)
```

---

### Step 8 — Model Saving

```python
model.save_pretrained("finetuned_model/final_model")
```

---

### Step 9 — Full Data Prediction

Function:

```python
predict_new_data()
```

Steps:

1. load trained model
2. load Reddit comments + submissions
3. merge data
4. batch prediction
5. save result

Output:

```
results/predicted_results.csv
```

---

### Step 10 — Topic Modeling (LDA)

Libraries:

- gensim
- nltk
- spacy

Steps:

1. tokenize text
2. remove stopwords
3. lemmatize
4. build corpus
5. run LDA

Output:

```
results/topics_top_words.csv
```

---

### Step 11 — Time Series Analysis

```python
group by date
→ daily count
→ weekly aggregation
```

Generate:

- shortage vs non-shortage trends
- percentage over time

---

### Step 12 — Drug-Level Analysis

Keyword matching:

```python
['Mounjaro', 'Wegovy', 'Ozempic', ...]
```

Then:

- plot time series per drug
- compare trends

---

## 8. Hyperparameters

| Parameter | Value |
|----------|------|
| Model | roberta-base |
| Batch size | 16 |
| Epochs | 20 |
| Learning rate | 2e-5 |
| Max length | 128 / 512 |
| Weight decay | 0.02 |
| Warmup steps | 500 |

---

## 9. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score (main metric)

---

## 10. How to Run

From root:

```bash
python Code/Project1_Encoder_Application/encoder_application/shortage_identification.py
```

---

## 11. Required Packages

Install:

```bash
pip install -r Code/requirements.txt
```

Key libraries:

- transformers
- torch
- pandas
- sklearn
- gensim
- nltk
- spacy

---

## 12. Important Notes

1. THIS PART Code is converted from Google Colab  
2. You MUST change paths:

```python
/content/drive/MyDrive/...
```
to local paths

3. Data is NOT included in repo
---

## 13. Outputs

## Model Storage

Due to GitHub file size limitations, the trained model is not included in this repository.

The model files (config, weights, tokenizer) are stored externally:

[https://drive.google.com/drive/folders/19qsqtcbE-7ideuKAzxKdw6gQhJSoFC01?usp=drive_link]

| file | description |
|------|------------|
| finetuned_model/ | trained model |
| predicted_results.csv | prediction output |
| updated_results.csv | enriched dataset |
| topics_top_words.csv | topic keywords |

---

## 14. Limitations

- depends on external data
- script is not modular
- label by person + AI
- LDA is exploratory

---

## 15. Future Work

- add XVAR and GNN


---

## 16. Project Position

This is **Project 1 (Encoder Application)**

Separate project:

```
Code/Project2_Decoder_Application/
```

## End