"""
Debug RAG Prompt - See what's being sent to LLM
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize
print("🔄 Loading...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cuda'}
)

vectordb = Chroma(
    persist_directory="/home/ubuntu/data/vector_db",
    embedding_function=embeddings
)

# Test question
question = "What is refractive surgery?"

# Retrieve documents
print(f"\n📝 Question: {question}\n")
print("🔍 Retrieving documents...")

docs = vectordb.similarity_search(question, k=2)

print(f"✅ Found {len(docs)} documents\n")

# Build context
context_parts = []
for i, doc in enumerate(docs, 1):
    print(f"📄 Document {i}:")
    print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"   Page: {doc.metadata.get('page', 'N/A')}")
    print(f"   Length: {len(doc.page_content)} chars")
    print(f"   Preview: {doc.page_content[:200]}...\n")
    
    context_parts.append(f"[Reference {i}]\n{doc.page_content[:500]}")

context = "\n\n".join(context_parts)

# Build prompt
prompt = f"""You are a professional medical assistant. Answer based on the provided medical references.

Medical References:
{context}

Question: {question}

Instructions:
1. Answer based on the references
2. Be clear and professional
3. Keep answer to 150-200 words

Answer:"""

print("="*80)
print("📋 FULL PROMPT TO LLM")
print("="*80)
print(prompt)
print("="*80)
print(f"\nPrompt length: {len(prompt)} characters")
print(f"Context length: {len(context)} characters\n")

# Test LLM
print("🔄 Testing LLM...")
llm = Ollama(model="deepseek-r1:1.5b")

import time
start = time.time()

response = llm.invoke(prompt)

elapsed = time.time() - start

print(f"⏱️  Time: {elapsed:.2f}s")
print(f"\n📝 Response ({len(response)} chars):")
print("="*80)
print(response)
print("="*80)
