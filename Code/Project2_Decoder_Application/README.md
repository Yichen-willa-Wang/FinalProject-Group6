# Clinical AI Agent - Medical RAG System

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system designed for medical follow-up assistance, specifically focused on ophthalmology and refractive surgery. The system combines a vector database of medical documents with large language models to provide accurate, contextual responses to patient questions.

## System Architecture

### Core Components

1. **Vector Database**: ChromaDB with 4,119 medical document chunks
2. **Embedding Model**: BAAI/bge-base-zh-v1.5 (768-dimensional, optimized for Chinese)
3. **LLM Options**: 
   - Local: DeepSeek-R1:1.5b via Ollama
   - API: ChatGPT (gpt-4o-mini)
4. **Framework**: LangChain for orchestration

### Data Pipeline

```
Medical Documents (800 files)
    ↓
Document Processing (PDF/Word splitting)
    ↓
Embedding Generation (GPU-accelerated)
    ↓
Vector Database (ChromaDB)
    ↓
RAG Query System
    ↓
Medical Follow-up Agent
```

## Directory Structure

```
clinical-ai-agent/
├── src/                          # Core source code
│   ├── rag_agent.py             # Main RAG agent implementation
│   ├── rag_agent_medical.py     # Medical follow-up specialized agent
│   ├── document_processor.py    # PDF/Word document processing
│   ├── pdf_processor.py         # PDF-specific handling
│   ├── query_decomposer.py      # Query decomposition utilities
│   ├── vector_db_builder.py     # Vector database construction
│   └── app_gradio.py            # Gradio web interface
│
├── scripts/                      # Utility scripts
│   ├── build_vectordb.py        # Build/rebuild vector database
│   ├── build_word_only.py       # Process Word documents only
│   ├── run_app.py               # Launch Gradio application
│   └── split_large_pdfs.py      # Split large PDF files
│
├── config/                       # Configuration files
│   └── config.yaml              # Main system configuration
│
├── test_scripts/                 # Evaluation and testing scripts
│   ├── evaluate_rag_final_v2.py        # Complete RAG evaluation (25 questions)
│   ├── evaluate_rag_show_answers.py    # Detailed answer display version
│   ├── debug_rag_prompt.py             # Debug prompt construction
│   ├── quick_test_rag.py               # Quick retrieval test
│   ├── test_medical_followup_deepseek.py  # Medical agent test
│   └── [other test scripts]
│
├── requirements.txt              # Python dependencies
└── rag_evaluation_*.xlsx        # Test results

Note: Vector database (~/data/vector_db/) not included due to size (54.6 MB)
```

## Technical Specifications

### Vector Database

- **Total Documents**: 800 files (784 PDFs + 16 Word documents)
- **Document Chunks**: 4,119 semantically segmented pieces
- **Total Characters**: 2,265,194
- **Database Size**: 54.6 MB
- **Average Chunks per File**: 45
- **Chunking Strategy**: Recursive character splitting with 800 char size, 100 char overlap

### Document Sources

Original medical literature split into manageable sizes:
- Original: 37 large PDF files (some over 500MB)
- Split Strategy: 10 pages per chunk, 1 page overlap
- Threshold: Files over 30 pages were split
- Final Dataset: All files under 7MB for safe processing

### Performance Metrics

**GPU-Accelerated (AWS A10G)**
- Embedding Speed: 10x faster than CPU
- Vector Retrieval: 0.011 seconds
- LLM Response (DeepSeek-R1): 10-20 seconds per query

**CPU-Only (Alibaba Cloud 2-core)**
- Vector Retrieval: ~1 second
- LLM Response (Ollama): 5 minutes per query

## Installation

### Prerequisites

- Python 3.9+
- CUDA 12.1+ (for GPU acceleration, optional)
- 16GB RAM minimum
- 20GB disk space

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys if using ChatGPT
```

3. Download or build vector database:
```bash
# If you have the medical documents
python scripts/build_vectordb.py

# Or download pre-built database (not included in this package)
```

## Usage

### Quick Start - Testing RAG Retrieval

Test the vector database retrieval without LLM:

```bash
python test_scripts/quick_test_rag.py
```

This validates:
- Vector database loading
- Embedding model functionality
- Document retrieval accuracy

### Running Full RAG Evaluation

Compare RAG performance with and without retrieved context:

```bash
python test_scripts/evaluate_rag_final_v2.py
```

This script:
- Tests 25 refractive surgery questions
- Compares "With RAG" vs "Without RAG" responses
- Generates Excel report with detailed metrics
- Shows prompt length, response time, and answer quality

### Medical Follow-up Agent

Test the empathetic medical response system:

```bash
python test_scripts/test_medical_followup_deepseek.py
```

Features:
- Role-based prompting (compassionate ophthalmologist)
- Emergency symptom detection
- Response length control (150-300 characters)
- Source citation

### Web Interface (Gradio)

Launch the interactive web UI:

```bash
python scripts/run_app.py
```

Access at `http://localhost:7860`

## Configuration

### config/config.yaml

Key parameters:

```yaml
paths:
  pdf_dir: /path/to/medical/pdfs
  vector_db_dir: /path/to/vector/db

models:
  embedding:
    name: BAAI/bge-base-zh-v1.5
  llm:
    name: deepseek-r1:1.5b  # or gpt-4o-mini

rag:
  chunk_size: 800
  chunk_overlap: 100
  top_k: 3
  similarity_threshold: 0.5
```

### LLM Options

**Option 1: Local Ollama (Free, Private)**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download model
ollama pull deepseek-r1:1.5b

# Start service
ollama serve
```

**Option 2: ChatGPT API (Fast, Paid)**

Set in `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
```

Then modify scripts to use `ChatOpenAI` instead of `Ollama`.

## Key Scripts Explained

### evaluate_rag_final_v2.py

**Purpose**: Comprehensive RAG system evaluation

**What it does**:
- Tests 25 questions about refractive surgery
- Runs each question twice: with RAG context and without
- Uses consistent medical system prompt for both conditions
- Generates Excel report comparing results

**Output**: Excel file with columns:
- Question
- Answer_NoRAG (general knowledge)
- Answer_WithRAG (retrieved context)
- Response times
- Prompt lengths
- Source citations

**Use case**: Measure RAG effectiveness, quality assurance

### debug_rag_prompt.py

**Purpose**: Inspect RAG prompt construction

**What it does**:
- Shows exactly what prompt is sent to LLM
- Displays retrieved documents and metadata
- Prints context length and source information
- Useful for debugging poor responses

**Use case**: Understanding why specific answers are generated

### Medical Agent Design

The `MedicalFollowupAgent` class uses a specialized prompt:

```python
system_role = """You are a compassionate and knowledgeable ophthalmologist. 
Your role is to:
- Answer patient questions with empathy and care
- Provide accurate, evidence-based medical information
- Use clear, professional language that patients can understand
- Address patient concerns and alleviate their worries
- Maintain clinical rigor while being approachable
- Always remind patients to consult their doctor for personalized advice
"""
```

Features:
- Response length control (150-300 characters)
- Emergency symptom detection (vision loss, severe pain, bleeding, fever)
- Avoids phrases like "based on the provided content"
- Replaces with "according to official guidelines"

## Evaluation Results

Based on testing 25 refractive surgery questions:

**Performance Comparison**:
- Average time (No RAG): ~2 seconds
- Average time (With RAG): ~4-5 seconds
- RAG overhead: ~2-3 seconds (retrieval + longer prompt)

**Quality Observations**:(see excel:rag_evaluation.xlsx)
- With RAG: More specific, cites actual medical sources
- Without RAG: More general, based on model's training data
- With RAG: Better for specialized questions (e.g., "Why does PRK recovery take so long?")
- Without RAG: Adequate for common questions (e.g., "What is refractive surgery?")

See `rag_evaluation_*.xlsx` for detailed results.

## Data Privacy and Deployment

### Local Deployment

- Use Ollama for complete data privacy
- All processing happens on your infrastructure
- No data sent to external APIs
- HIPAA compliance-friendly

### API Deployment

- Review data privacy agreements
- Consider PHI/PII implications
- Implement appropriate data handling

## Troubleshooting

### Vector Database Issues

**Problem**: ChromaDB fails to load

**Solution**: 
```bash
# Rebuild database
python scripts/build_vectordb.py
```

### GPU Not Detected

**Problem**: Embeddings running on CPU despite GPU availability

**Solution**:
```python
# In code, explicitly set device
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cuda'}  # Force CUDA
)
```

### Ollama Connection Errors

**Problem**: "404 Not Found" from Ollama

**Solution**:
```bash
# Ensure Ollama is running
ollama serve &

# Verify model is downloaded
ollama list

# Test connection
curl http://localhost:11434/api/tags
```

### Slow Performance

**Problem**: Queries taking too long

**Solutions**:
1. Reduce `top_k` in config (fewer documents retrieved)
2. Shorten context length (max 300 chars per document)
3. Use smaller LLM model (e.g., deepseek-r1:1.5b instead of 7b)
4. Enable GPU acceleration for embeddings

## Development Notes

### Adding New Medical Documents

1. Place PDFs in configured directory
2. Run vector database builder:
```bash
python scripts/build_vectordb.py
```

3. Verify chunk count increased:
```bash
python quick_test_rag.py
```

### Customizing Prompts

Edit `src/rag_agent_medical.py`:

```python
class MedicalFollowupAgent:
    def __init__(self, ...):
        self.system_role = """[Your custom prompt here]"""
```

### Modifying Retrieval Parameters

In `config/config.yaml`:
```yaml
rag:
  top_k: 5              # Number of documents to retrieve
  similarity_threshold: 0.5  # Minimum similarity score (0-1)
```

## Testing Strategy

The project includes multiple levels of testing:

1. **Unit Tests**: Quick retrieval validation (quick_test_rag.py)
2. **Integration Tests**: Full RAG pipeline (debug_rag_prompt.py)
3. **Evaluation Tests**: 25-question benchmark (evaluate_rag_final_v2.py)
4. **Medical Agent Tests**: Specialized prompting (test_medical_followup_deepseek.py)

## Known Limitations

1. **Vector database not included**: Too large for package (54.6 MB)
2. **Chinese-optimized**: Embedding model designed for Chinese, works for English but not optimal
3. **Medical domain specific**: Trained on ophthalmology literature, may not generalize
4. **Context window**: Limited to ~2000 characters to avoid LLM timeouts
5. **No conversation memory**: Each query is independent

## Future Enhancements

Potential improvements:
- Add conversation history tracking
- Implement query decomposition for complex questions
- Multi-modal support (images, diagrams)
- Fine-tune LLM on medical literature
- Add evaluation metrics (BLEU, ROUGE, medical accuracy)
- Implement caching for frequent queries

## Citation and Acknowledgments

### Data Sources
- Medical documents from ophthalmology textbooks and clinical guidelines
- Refractive surgery PPP (Preferred Practice Pattern) documents

### Models Used
- Embedding: BAAI/bge-base-zh-v1.5 (Beijing Academy of Artificial Intelligence)
- LLM: localize DeepSeek-R1 (DeepSeek AI)(1.5G)
- Alternative LLM: GPT-4o-mini (OpenAI)

### Frameworks
- LangChain: RAG orchestration
- ChromaDB: Vector database
- Gradio: Web interface
- Ollama: Local LLM serving

## Contact and Support

For questions about implementation or deployment:
1. Review test scripts for usage examples
2. Check config.yaml for parameter tuning
3. Consult evaluation Excel files for expected performance

## License

This project uses medical literature that may be subject to copyright. Ensure proper licensing before deployment in production environments.

---

**Last Updated**: 2026-04-29  
**Version**: 1.0  
**Tested Environment**: Ubuntu 24.04, Python 3.9, CUDA 12.1