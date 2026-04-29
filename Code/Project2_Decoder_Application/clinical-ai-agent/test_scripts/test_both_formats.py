#!/usr/bin/env python3
"""Test both PDF and Word processing with detailed output"""

import sys
sys.path.append('src')

from document_processor import DocumentProcessor
from vector_db_builder import HybridChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

doc_processor = DocumentProcessor()
pdf_dir = Path("/root/data/raw/medical_pdfs")

print("="*80)
print("DETAILED DOCUMENT PROCESSING TEST")
print("="*80)

# Test PDF - Show more details
print("\n" + "="*80)
print("PART 1: PDF PROCESSING")
print("="*80)
pdf_files = list(pdf_dir.glob("*.pdf"))[:3]  # Test first 3 PDFs

for idx, pdf_file in enumerate(pdf_files, 1):
    print(f"\n--- PDF {idx}/{len(pdf_files)}: {pdf_file.name} ---")
    print(f"File size: {pdf_file.stat().st_size / 1024:.1f} KB")
    
    pdf_data = doc_processor.process_document(str(pdf_file))
    
    print(f"\nExtracted content:")
    print(f"  • Text pages: {len(pdf_data['text_content'])}")
    print(f"  • Tables: {len(pdf_data['tables'])}")
    print(f"  • Images: {len(pdf_data['images'])}")
    
    # Show first page details
    if pdf_data['text_content']:
        first_page = pdf_data['text_content'][0]
        print(f"\nFirst page details:")
        print(f"  • Page number: {first_page['page']}")
        print(f"  • Type: {first_page['type']}")
        print(f"  • Character count: {first_page['char_count']}")
        print(f"\n  Content preview (first 400 chars):")
        print("  " + "-"*76)
        for line in first_page['text'][:400].split('\n')[:10]:
            print(f"  {line}")
        print("  " + "-"*76)
    
    # Show table details if any
    if pdf_data['tables']:
        print(f"\nTable details:")
        for i, table in enumerate(pdf_data['tables'][:2], 1):
            print(f"\n  Table {i}:")
            print(f"    • Page: {table['page']}")
            print(f"    • Size: {table['rows']} rows × {table['columns']} columns")
            print(f"    • Columns: {', '.join(table['dataframe'].columns)}")
            print(f"\n    Text representation (first 300 chars):")
            print("    " + table['text_representation'][:300])

# Test Word - Show more details
print("\n\n" + "="*80)
print("PART 2: WORD DOCUMENT PROCESSING")
print("="*80)
docx_files = list(pdf_dir.glob("*.docx"))[:3]  # Test first 3 Word files

for idx, docx_file in enumerate(docx_files, 1):
    print(f"\n--- Word {idx}/{len(docx_files)}: {docx_file.name} ---")
    print(f"File size: {docx_file.stat().st_size / 1024:.1f} KB")
    
    docx_data = doc_processor.process_document(str(docx_file))
    
    print(f"\nExtracted content:")
    print(f"  • Text sections: {len(docx_data['text_content'])}")
    print(f"  • Tables: {len(docx_data['tables'])}")
    
    if docx_data['text_content']:
        text_content = docx_data['text_content'][0]
        print(f"\nText details:")
        print(f"  • Total characters: {text_content['char_count']}")
        print(f"  • Paragraphs (approx): {text_content['text'].count(chr(10)) + 1}")
        print(f"\n  Content preview (first 500 chars):")
        print("  " + "-"*76)
        for line in text_content['text'][:500].split('\n')[:15]:
            print(f"  {line}")
        print("  " + "-"*76)
    
    if docx_data['tables']:
        print(f"\nTable details:")
        for i, table in enumerate(docx_data['tables'][:2], 1):
            print(f"\n  Table {i}:")
            print(f"    • Size: {table['rows']} rows × {table['columns']} columns")
            print(f"    • Text representation:\n    {table['text_representation'][:300]}")

# Test Chunking with details
print("\n\n" + "="*80)
print("PART 3: CHUNKING ANALYSIS")
print("="*80)

print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Creating chunker (chunk_size=800, overlap=100)...")
chunker = HybridChunker(
    embeddings=embeddings,
    chunk_size=800,
    chunk_overlap=100,
    use_semantic=False
)

# Test chunking on first PDF
if pdf_files:
    test_file = pdf_files[0]
    print(f"\nTesting chunking on: {test_file.name}")
    
    doc_data = doc_processor.process_document(str(test_file))
    doc_data['source'] = str(test_file)
    chunks = chunker.chunk_document(doc_data)
    
    print(f"\n✓ Generated {len(chunks)} chunks")
    
    # Chunk statistics
    text_chunks = [c for c in chunks if c.metadata.get('content_type') == 'text']
    table_chunks = [c for c in chunks if c.metadata.get('content_type') == 'table']
    
    print(f"\nChunk breakdown:")
    print(f"  • Text chunks: {len(text_chunks)}")
    print(f"  • Table chunks: {len(table_chunks)}")
    
    # Size distribution
    sizes = [len(c.page_content) for c in chunks]
    print(f"\nChunk size distribution:")
    print(f"  • Minimum: {min(sizes)} characters")
    print(f"  • Maximum: {max(sizes)} characters")
    print(f"  • Average: {sum(sizes)//len(sizes)} characters")
    print(f"  • Median: {sorted(sizes)[len(sizes)//2]} characters")
    
    # Show first 5 chunks in detail
    print(f"\n" + "-"*80)
    print("FIRST 5 CHUNKS (DETAILED):")
    print("-"*80)
    
    for i, chunk in enumerate(chunks[:5], 1):
        print(f"\n┌─ Chunk {i} " + "─"*68)
        print(f"│ Metadata:")
        print(f"│   • Source: {chunk.metadata.get('source', 'N/A')}")
        print(f"│   • Page: {chunk.metadata.get('page', 'N/A')}")
        print(f"│   • Type: {chunk.metadata.get('content_type', 'N/A')}")
        print(f"│   • Chunk ID: {chunk.metadata.get('chunk_id', 'N/A')}")
        print(f"│   • Length: {len(chunk.page_content)} chars")
        print(f"│")
        print(f"│ Content:")
        print(f"│ " + "─"*76)
        for line in chunk.page_content[:400].split('\n')[:12]:
            print(f"│ {line[:74]}")
        if len(chunk.page_content) > 400:
            print(f"│ ... (+ {len(chunk.page_content)-400} more chars)")
        print(f"└" + "─"*78)

# Test chunking on first Word file
if docx_files:
    test_file = docx_files[0]
    print(f"\n\nTesting chunking on Word: {test_file.name}")
    
    doc_data = doc_processor.process_document(str(test_file))
    doc_data['source'] = str(test_file)
    chunks = chunker.chunk_document(doc_data)
    
    print(f"\n✓ Generated {len(chunks)} chunks from Word document")
    
    print(f"\n" + "-"*80)
    print("FIRST 3 CHUNKS FROM WORD:")
    print("-"*80)
    
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n┌─ Chunk {i} " + "─"*68)
        print(f"│ Length: {len(chunk.page_content)} chars")
        print(f"│ Content:")
        print(f"│ " + "─"*76)
        for line in chunk.page_content[:300].split('\n')[:10]:
            print(f"│ {line[:74]}")
        print(f"└" + "─"*78)

print("\n" + "="*80)
print("✓ DETAILED TESTING COMPLETED")
print("="*80)
