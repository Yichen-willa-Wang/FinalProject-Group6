#!/usr/bin/env python3
"""
Test chunking on a single document
"""

import sys
sys.path.append('src')

from document_processor import DocumentProcessor
from vector_db_builder import VectorDBBuilder, HybridChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Pick one PDF to test
test_file = "/root/data/raw/medical_pdfs/术前注意事项_表层.docx"

print("="*60)
print("Testing Chunking on Single Document")
print("="*60)
print(f"File: {test_file}")
print("="*60)

# Initialize processors
doc_processor = DocumentProcessor()
print("\n1. Processing document...")
doc_data = doc_processor.process_document(test_file)

print(f"\nExtracted content:")
print(f"  - Text items: {len(doc_data['text_content'])}")
print(f"  - Tables: {len(doc_data['tables'])}")
print(f"  - Images: {len(doc_data['images'])}")

# Show first text content
if doc_data['text_content']:
    first_text = doc_data['text_content'][0]['text']
    print(f"\nFirst 500 characters of text:")
    print("-"*60)
    print(first_text[:500])
    print("-"*60)

# Initialize embeddings and chunker
print("\n2. Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("\n3. Creating chunks...")
chunker = HybridChunker(
    embeddings=embeddings,
    chunk_size=800,
    chunk_overlap=100,
    use_semantic=True
)

doc_data['source'] = test_file
chunks = chunker.chunk_document(doc_data)

print(f"\nGenerated {len(chunks)} chunks")
print("="*60)

# Show first 3 chunks
print("\nFirst 3 chunks:")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\n{'='*60}")
    print(f"Chunk {i}:")
    print(f"  Type: {chunk.metadata.get('content_type', 'unknown')}")
    print(f"  Page: {chunk.metadata.get('page', 'unknown')}")
    print(f"  Length: {len(chunk.page_content)} chars")
    print(f"\nContent:")
    print("-"*60)
    print(chunk.page_content)
    print("-"*60)

# Show table chunks if any
table_chunks = [c for c in chunks if c.metadata.get('content_type') == 'table']
if table_chunks:
    print(f"\n{'='*60}")
    print(f"Found {len(table_chunks)} table chunks")
    print(f"{'='*60}")
    print("\nFirst table chunk:")
    print("-"*60)
    print(table_chunks[0].page_content)
    print("-"*60)

# Statistics
print(f"\n{'='*60}")
print("Chunking Statistics:")
print(f"{'='*60}")
print(f"Total chunks: {len(chunks)}")
print(f"Text chunks: {sum(1 for c in chunks if c.metadata.get('content_type') == 'text')}")
print(f"Table chunks: {sum(1 for c in chunks if c.metadata.get('content_type') == 'table')}")
print(f"Image refs: {sum(1 for c in chunks if c.metadata.get('content_type') == 'image')}")

# Chunk size distribution
sizes = [len(c.page_content) for c in chunks]
print(f"\nChunk size distribution:")
print(f"  Min: {min(sizes)} chars")
print(f"  Max: {max(sizes)} chars")
print(f"  Average: {sum(sizes)//len(sizes)} chars")

print("\n✓ Chunking test completed!")
