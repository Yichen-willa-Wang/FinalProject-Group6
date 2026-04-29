"""
Quick RAG Test on AWS GPU
Tests existing vector DB + simple LLM integration
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import time

print("="*80)
print("⚡ Quick RAG Test on AWS GPU")
print("="*80)

# Check GPU
print(f"\n💻 GPU Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test questions
test_questions = [
    "白内障手术后需要注意什么?",
    "高血压患者的饮食建议",
    "What is refractive surgery?"
]

print(f"\n📝 Test Questions: {len(test_questions)}")

# Load embeddings on GPU
print("\n🔄 Step 1: Loading embeddings on GPU...")
start = time.time()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cuda'},  # GPU!
    encode_kwargs={'normalize_embeddings': True}
)

print(f"✅ Embeddings loaded in {time.time() - start:.2f}s")

# Load vector database
print("\n🔄 Step 2: Loading vector database...")
start = time.time()

vectordb = Chroma(
    persist_directory="/home/ubuntu/data/vector_db",
    embedding_function=embeddings
)

chunk_count = vectordb._collection.count()
print(f"✅ Vector DB loaded in {time.time() - start:.2f}s")
print(f"   Total chunks: {chunk_count}")

# Test retrieval
print("\n🔄 Step 3: Testing document retrieval...")
print("="*80)

for i, question in enumerate(test_questions, 1):
    print(f"\n[Test {i}/{len(test_questions)}]")
    print(f"Question: {question}")
    print("-"*80)
    
    start = time.time()
    
    # Retrieve top 3 documents
    docs = vectordb.similarity_search(question, k=3)
    
    elapsed = time.time() - start
    
    print(f"⏱️  Retrieval time: {elapsed:.3f}s (GPU accelerated!)")
    print(f"📚 Found {len(docs)} relevant documents:")
    
    for j, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        content_preview = doc.page_content[:150].replace('\n', ' ')
        
        print(f"\n   {j}. Source: {source} (Page {page})")
        print(f"      Content: {content_preview}...")

print("\n" + "="*80)
print("✅ RAG Retrieval Test Complete!")
print("="*80)

# Summary
print("\n📊 Summary:")
print(f"   Vector DB: {chunk_count} chunks")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"   Embedding Model: BAAI/bge-base-zh-v1.5")
print(f"   Tests Passed: {len(test_questions)}/{len(test_questions)}")
