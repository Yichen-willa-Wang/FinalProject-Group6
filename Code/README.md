# Code Overview

This folder contains two independent NLP business application projects.

The first project applies an encoder-based classification model to detect drug shortage-related discussions from Reddit data. The second project applies a decoder-based RAG agent to support medical follow-up question answering.

These two projects are independent and are not implemented as a single encoder–decoder pipeline.

---

## Project 1: Encoder Application — Social Media Shortage Detection

This project uses a transformer-based text classifier to identify shortage-related discussions from Reddit posts and comments about GLP-1 medications.

### Main Functions

- Clean and label Reddit text data
- Train a transformer-based binary classifier
- Predict shortage-related posts in the full corpus
- Generate structured shortage signals
- Support topic modeling and time-series analysis

### Main Script

```text
Code/Project1_Encoder_Application/encoder_application/shortage_identification.py
````

### Project README

```text
Code/Project1_Encoder_Application/README.md
```

### Dependencies

```text
Code/Project1_Encoder_Application/requirements.txt
```

---

## Project 2: Decoder Application — Clinical AI Agent

This project implements a medical Retrieval-Augmented Generation (RAG) agent for ophthalmology and refractive-surgery follow-up assistance.

### Main Functions

* Process medical documents
* Build a ChromaDB vector database
* Retrieve relevant clinical document chunks
* Generate patient-facing responses using an LLM
* Evaluate RAG responses against non-RAG responses

### Main Folder

```text
Code/Project2_Decoder_Application/clinical-ai-agent/
```

### Main Scripts

```text
Code/Project2_Decoder_Application/clinical-ai-agent/scripts/build_vectordb.py
Code/Project2_Decoder_Application/clinical-ai-agent/test_scripts/quick_test_rag.py
Code/Project2_Decoder_Application/clinical-ai-agent/test_scripts/evaluate_rag_final.py
Code/Project2_Decoder_Application/clinical-ai-agent/scripts/run_app.py
```

### Project README

```text
Code/Project2_Decoder_Application/README.md
```

---

## How to Run

### Project 1

Install dependencies:

```bash
pip install -r Code/Project1_Encoder_Application/requirements.txt
```

Run the encoder application:

```bash
python Code/Project1_Encoder_Application/encoder_application/shortage_identification.py
```

### Project 2

Install dependencies:

```bash
pip install -r Code/Project2_Decoder_Application/clinical-ai-agent/requirements.txt
```

Run a quick RAG test:

```bash
python Code/Project2_Decoder_Application/clinical-ai-agent/test_scripts/quick_test_rag.py
```

Run the RAG evaluation:

```bash
python Code/Project2_Decoder_Application/clinical-ai-agent/test_scripts/evaluate_rag_final.py
```

Launch the Gradio interface:

```bash
python Code/Project2_Decoder_Application/clinical-ai-agent/scripts/run_app.py
```

---

## Data and Model Notes

Large datasets, trained model checkpoints, and vector databases are not included in this repository due to size limitations.

Project-specific data and model details are documented in each project README.

---

## End

