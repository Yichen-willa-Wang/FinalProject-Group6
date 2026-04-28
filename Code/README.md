# Code Overview

This repository contains two independent NLP projects, each exploring a different business application.

---

## Project 1: Encoder Application — Social Media Shortage Detection

This project focuses on detecting drug shortage-related discussions from Reddit data using a transformer-based classification model.

### Key Functions

- Transform unstructured Reddit text into structured shortage signals
- Identify shortage-related discussions
- Support downstream analysis such as:
  - time-series analysis
  - topic modeling
  - medication-level trends

### Main Script

Code/Project1_Encoder_Application/encoder_application/shortage_identification.py


---

## Project 2: Decoder Application — LLM Agent

This project explores a separate business application using LLM-based agents.

It focuses on:

- generating responses or decisions
- simulating agent behavior
- interacting with structured or unstructured inputs

⚠️ This project is currently under development.

---

## Important Note

These two projects are **independent**:

- Project 1 focuses on encoder-based classification
- Project 2 focuses on LLM/agent-based generation

They are not implemented as a single encoder–decoder pipeline.

---

## How to Run

### Step 1: Install dependencies
pip install -r Code/requirements.txt
### Step 2: Run Encoder Project
python Code/Project1_Encoder_Application/encoder_application/shortage_identification.py


---

## Data and Model

- Large datasets are not included due to limitations
- Trained models are stored externally (see project README)

---

## End