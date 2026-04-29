"""
RAG Evaluation: With RAG vs Without RAG
Tests 25 refractive surgery questions
Generates Excel comparison report
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("="*80)
print("📊 RAG Evaluation: With RAG vs Without RAG")
print("="*80)

# ============================================================================
# Test Questions - 25 Refractive Surgery FAQs
# ============================================================================

questions = [
    "What is refractive surgery?",
    "Can I have both eyes treated at the same time?",
    "Who can have the refractive surgery?",
    "What refractive defects can refractive surgery correct?",
    "Will the refractive surgery patient see much better than with glasses or contact lenses?",
    "What are the most common complications?",
    "Is any pain felt after laser surgery?",
    "Can a person have a cataract operation after laser surgery?",
    "If the patient has already undergone refractive surgery, is it possible to undergo surgery again in the future?",
    "Which technique is better: laser refractive surgery or intraocular lens implantation?",
    "Is it possible to undergo refractive surgery if other visual problems are present?",
    "What is achieved with laser surgery?",
    "Is refractive surgery covered by insurance?",
    "How safe is refractive surgery?",
    "What if I blink or move during the procedure?",
    "How soon after the surgery will I be able to see?",
    "How often will I see the doctor following my surgery?",
    "How is PRK or LASIK likely to affect my need to use glasses or contacts when I get older?",
    "Can I still drive at night after refractive surgery?",
    "Does laser eye surgery hurt?",
    "Is there a suitable alternative if I have thin corneas?",
    "What is the difference between PRK and LASIK surgery?",
    "Am I a suitable candidate for laser eye surgery?",
    "Does SMILE surgery differ from LASIK and PRK surgery?",
    "Why does PRK recovery take so long?"
]

print(f"\n📝 Total questions: {len(questions)}")

# ============================================================================
# Ollama LLM Wrapper
# ============================================================================

class OllamaLLM:
    """Ollama local LLM wrapper"""
    
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt: str, max_tokens=300) -> str:
        """Generate response using Ollama"""
        import requests
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(self.api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            raise Exception(f"Ollama error: {response.status_code}")

# ============================================================================
# Setup
# ============================================================================

print("\n🔄 Initializing components...")

# GPU check
print(f"💻 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Initialize LLM
llm = OllamaLLM()

# Test LLM
try:
    test = llm.generate("Hello", max_tokens=10)
    print(f"✅ LLM ready: {test[:30]}...")
except Exception as e:
    print(f"❌ LLM error: {e}")
    sys.exit(1)

# Load embeddings
print("\n🔄 Loading embeddings on GPU...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cuda'}
)

# Load vector database
print("🔄 Loading vector database...")
vectordb = Chroma(
    persist_directory="/home/ubuntu/data/vector_db",
    embedding_function=embeddings
)

chunk_count = vectordb._collection.count()
print(f"✅ Vector DB: {chunk_count} chunks\n")

# ============================================================================
# Test Functions
# ============================================================================

def test_without_rag(question: str) -> Dict:
    """Test LLM without RAG (no context)"""
    
    prompt = f"""You are a medical assistant. Answer this question about refractive surgery.

Question: {question}

Provide a clear, professional answer (150 words):"""
    
    start = time.time()
    try:
        answer = llm.generate(prompt, max_tokens=200)
        elapsed = time.time() - start
        
        return {
            'answer': answer,
            'time': elapsed,
            'sources': '',
            'status': 'Success'
        }
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'time': time.time() - start,
            'sources': '',
            'status': 'Error'
        }

def test_with_rag(question: str) -> Dict:
    """Test LLM with RAG (retrieval + context)"""
    
    # Retrieve relevant documents
    start = time.time()
    
    try:
        docs = vectordb.similarity_search(question, k=2)
        
        if not docs:
            return test_without_rag(question)  # Fallback
        
        # Build context
        context = "\n\n".join([f"[Source {i+1}]\n{doc.page_content[:300]}" 
                               for i, doc in enumerate(docs)])
        
        # Build prompt with context
        prompt = f"""You are a medical assistant. Answer the question based on the provided medical knowledge.

Medical Knowledge:
{context}

Question: {question}

Provide a clear, professional answer based on the above knowledge (150 words):"""
        
        answer = llm.generate(prompt, max_tokens=200)
        elapsed = time.time() - start
        
        # Extract sources
        sources = '; '.join([f"{doc.metadata.get('source', 'Unknown')} (p.{doc.metadata.get('page', 'N/A')})" 
                            for doc in docs])
        
        return {
            'answer': answer,
            'time': elapsed,
            'sources': sources,
            'status': 'Success'
        }
        
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'time': time.time() - start,
            'sources': '',
            'status': 'Error'
        }

# ============================================================================
# Run Tests
# ============================================================================

print("="*80)
print("🧪 Running Evaluation Tests")
print("="*80 + "\n")

results = []

for i, question in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] {question[:60]}...")
    print("-"*80)
    
    # Test WITHOUT RAG
    print("  1️⃣ Testing WITHOUT RAG...")
    no_rag_result = test_without_rag(question)
    print(f"     ⏱️  {no_rag_result['time']:.2f}s")
    
    # Test WITH RAG
    print("  2️⃣ Testing WITH RAG...")
    with_rag_result = test_with_rag(question)
    print(f"     ⏱️  {with_rag_result['time']:.2f}s")
    print(f"     📚 Sources: {with_rag_result['sources'][:50]}...")
    
    print()
    
    # Store results
    results.append({
        'Question_ID': i,
        'Question': question,
        
        # Without RAG
        'Answer_No_RAG': no_rag_result['answer'],
        'Time_No_RAG_Sec': round(no_rag_result['time'], 2),
        'Status_No_RAG': no_rag_result['status'],
        
        # With RAG
        'Answer_With_RAG': with_rag_result['answer'],
        'Time_With_RAG_Sec': round(with_rag_result['time'], 2),
        'Sources_With_RAG': with_rag_result['sources'],
        'Status_With_RAG': with_rag_result['status'],
        
        # Comparison
        'Time_Difference_Sec': round(with_rag_result['time'] - no_rag_result['time'], 2),
        'Answer_Length_No_RAG': len(no_rag_result['answer']),
        'Answer_Length_With_RAG': len(with_rag_result['answer'])
    })

# ============================================================================
# Generate Excel Report
# ============================================================================

print("="*80)
print("📊 Generating Excel Report")
print("="*80)

df = pd.DataFrame(results)

# Calculate statistics
total_time_no_rag = df['Time_No_RAG_Sec'].sum()
total_time_with_rag = df['Time_With_RAG_Sec'].sum()
avg_time_no_rag = df['Time_No_RAG_Sec'].mean()
avg_time_with_rag = df['Time_With_RAG_Sec'].mean()

success_no_rag = len(df[df['Status_No_RAG'] == 'Success'])
success_with_rag = len(df[df['Status_With_RAG'] == 'Success'])

# Create Excel with multiple sheets
output_filename = f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # Sheet 1: Full Results
    df.to_excel(writer, sheet_name='Full Results', index=False)
    
    ws1 = writer.sheets['Full Results']
    ws1.column_dimensions['A'].width = 10  # ID
    ws1.column_dimensions['B'].width = 70  # Question
    ws1.column_dimensions['C'].width = 80  # Answer No RAG
    ws1.column_dimensions['D'].width = 15  # Time No RAG
    ws1.column_dimensions['E'].width = 15  # Status No RAG
    ws1.column_dimensions['F'].width = 80  # Answer With RAG
    ws1.column_dimensions['G'].width = 15  # Time With RAG
    ws1.column_dimensions['H'].width = 60  # Sources
    ws1.column_dimensions['I'].width = 15  # Status With RAG
    
    # Text wrapping
    from openpyxl.styles import Alignment, Font
    for row in ws1.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    
    for cell in ws1[1]:
        cell.font = Font(bold=True)
    
    # Sheet 2: Summary Statistics
    summary_data = {
        'Metric': [
            'Total Questions',
            'Success Rate - No RAG (%)',
            'Success Rate - With RAG (%)',
            'Average Time - No RAG (s)',
            'Average Time - With RAG (s)',
            'Total Time - No RAG (s)',
            'Total Time - With RAG (s)',
            'Time Overhead - RAG (s)',
            'Average Answer Length - No RAG',
            'Average Answer Length - With RAG',
            'Model Used',
            'Vector DB Chunks',
            'GPU',
            'Test Date'
        ],
        'Value': [
            len(questions),
            round(success_no_rag / len(questions) * 100, 1),
            round(success_with_rag / len(questions) * 100, 1),
            round(avg_time_no_rag, 2),
            round(avg_time_with_rag, 2),
            round(total_time_no_rag, 2),
            round(total_time_with_rag, 2),
            round(total_time_with_rag - total_time_no_rag, 2),
            round(df['Answer_Length_No_RAG'].mean(), 0),
            round(df['Answer_Length_With_RAG'].mean(), 0),
            'DeepSeek-R1:1.5b',
            chunk_count,
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    ws2 = writer.sheets['Summary']
    ws2.column_dimensions['A'].width = 40
    ws2.column_dimensions['B'].width = 30
    
    # Sheet 3: Comparison (side by side)
    comparison_df = df[['Question_ID', 'Question', 'Answer_No_RAG', 'Answer_With_RAG', 
                        'Time_No_RAG_Sec', 'Time_With_RAG_Sec', 'Sources_With_RAG']]
    comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
    
    ws3 = writer.sheets['Comparison']
    ws3.column_dimensions['B'].width = 70
    ws3.column_dimensions['C'].width = 80
    ws3.column_dimensions['D'].width = 80
    ws3.column_dimensions['G'].width = 60

print(f"\n✅ Excel report saved: {output_filename}\n")

# ============================================================================
# Console Summary
# ============================================================================

print("="*80)
print("📈 EVALUATION SUMMARY")
print("="*80)

print(f"\n📊 Overall Results:")
print(f"   Total Questions:        {len(questions)}")
print(f"   Success (No RAG):       {success_no_rag}/{len(questions)} ({success_no_rag/len(questions)*100:.1f}%)")
print(f"   Success (With RAG):     {success_with_rag}/{len(questions)} ({success_with_rag/len(questions)*100:.1f}%)")

print(f"\n⏱️  Performance:")
print(f"   Avg Time (No RAG):      {avg_time_no_rag:.2f}s")
print(f"   Avg Time (With RAG):    {avg_time_with_rag:.2f}s")
print(f"   RAG Overhead:           +{avg_time_with_rag - avg_time_no_rag:.2f}s per question")
print(f"   Total Time (No RAG):    {total_time_no_rag:.2f}s ({total_time_no_rag/60:.1f} min)")
print(f"   Total Time (With RAG):  {total_time_with_rag:.2f}s ({total_time_with_rag/60:.1f} min)")

print(f"\n📝 Answer Quality:")
print(f"   Avg Length (No RAG):    {df['Answer_Length_No_RAG'].mean():.0f} chars")
print(f"   Avg Length (With RAG):  {df['Answer_Length_With_RAG'].mean():.0f} chars")

print(f"\n🔧 Configuration:")
print(f"   Model:                  DeepSeek-R1:1.5b")
print(f"   GPU:                    {torch.cuda.get_device_name(0)}")
print(f"   Vector DB:              {chunk_count} chunks")
print(f"   Top-K Retrieval:        2")

print(f"\n📄 Output: {output_filename}")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
