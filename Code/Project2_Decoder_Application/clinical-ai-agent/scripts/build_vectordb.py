#!/usr/bin/env python3
"""
Build Vector Database with detailed progress tracking
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_processor import DocumentProcessor
from vector_db_builder import VectorDBBuilder, HybridChunker
from langchain.docstore.document import Document
import argparse
import yaml
import time
from datetime import datetime
from tqdm import tqdm

def load_config(config_path="config/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def format_time(seconds):
    """Format seconds to readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Build vector database from documents')
    parser.add_argument('--pdf_folder', type=str, 
                       help='Path to PDF/Word folder (default: from config)')
    parser.add_argument('--output', type=str,
                       help='Output path for vector database (default: from config)')
    parser.add_argument('--embedding_model', type=str,
                       help='Embedding model name (default: from config)')
    parser.add_argument('--chunk_size', type=int, default=800,
                       help='Chunk size (default: 800)')
    parser.add_argument('--chunk_overlap', type=int, default=100,
                       help='Chunk overlap (default: 100)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Set paths from config or args
    pdf_folder = args.pdf_folder or config['paths']['pdf_dir']
    output_path = args.output or config['paths']['vector_db_dir']
    embedding_model = args.embedding_model or config['models']['embedding']['name']
    
    print("="*80)
    print("VECTOR DATABASE BUILDER WITH DETAILED PROGRESS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  • Input folder: {pdf_folder}")
    print(f"  • Output path: {output_path}")
    print(f"  • Embedding model: {embedding_model}")
    print(f"  • Chunk size: {args.chunk_size}")
    print(f"  • Chunk overlap: {args.chunk_overlap}")
    print("="*80)
    
    # Initialize processors
    print("\n[1/5] Initializing processors...")
    doc_processor = DocumentProcessor()
    db_builder = VectorDBBuilder(model_name=embedding_model, db_type="chroma")
    print("  ✓ Document processor ready")
    print("  ✓ Embedding model loaded")
    
    # Initialize hybrid chunker
    print("\n[2/5] Initializing chunker...")
    chunker = HybridChunker(
        embeddings=db_builder.embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic=False
    )
    print("  ✓ Chunker ready")
    
    # Scan documents
    print("\n[3/5] Scanning documents...")
    pdf_folder_path = Path(pdf_folder)
    
    pdf_files = list(pdf_folder_path.glob("*.pdf"))
    docx_files = list(pdf_folder_path.glob("*.docx"))
    doc_files = list(pdf_folder_path.glob("*.doc"))
    
    all_files = pdf_files + docx_files + doc_files
    
    print(f"  ✓ Found {len(all_files)} documents:")
    print(f"    - PDF files: {len(pdf_files)}")
    print(f"    - Word files (.docx): {len(docx_files)}")
    print(f"    - Word files (.doc): {len(doc_files)}")
    
    if len(all_files) == 0:
        print("\n✗ No documents found!")
        return
    
    # Process all documents
    print("\n[4/5] Processing documents...")
    print("="*80)
    
    all_chunks = []
    processing_stats = {
        'total_files': len(all_files),
        'processed': 0,
        'failed': 0,
        'total_chunks': 0,
        'total_chars': 0
    }
    
    # Use tqdm for overall progress
    with tqdm(total=len(all_files), desc="Overall Progress", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              position=0) as pbar_overall:
        
        for idx, doc_file in enumerate(all_files, 1):
            file_start_time = time.time()
            
            # Update main progress bar description
            pbar_overall.set_description(f"Processing [{idx}/{len(all_files)}] {doc_file.name[:40]}")
            
            print(f"\n{'='*80}")
            print(f"File {idx}/{len(all_files)}: {doc_file.name}")
            print(f"  Size: {doc_file.stat().st_size / 1024:.1f} KB | Type: {doc_file.suffix.upper()}")
            print(f"{'='*80}")
            
            try:
                # Extract content with progress
                print("  [1/2] Extracting content...")
                doc_data = doc_processor.process_document(str(doc_file))
                doc_data['source'] = doc_file.name
                
                text_items = len(doc_data.get('text_content', []))
                tables = len(doc_data.get('tables', []))
                images = len(doc_data.get('images', []))
                
                print(f"        ✓ Extracted: {text_items} pages, {tables} tables, {images} images")
                
                # Chunk the document with progress
                print("  [2/2] Chunking...")
                chunks = chunker.chunk_document(doc_data)
                
                text_chunks = sum(1 for c in chunks if c.metadata.get('content_type') == 'text')
                table_chunks = sum(1 for c in chunks if c.metadata.get('content_type') == 'table')
                total_chars = sum(len(c.page_content) for c in chunks)
                
                print(f"        ✓ Generated {len(chunks)} chunks ({text_chunks} text, {table_chunks} tables)")
                
                chunk_sizes = [len(c.page_content) for c in chunks]
                if chunk_sizes:
                    avg_size = sum(chunk_sizes)//len(chunk_sizes)
                    print(f"        ✓ Size: avg={avg_size}, min={min(chunk_sizes)}, max={max(chunk_sizes)} chars")
                
                all_chunks.extend(chunks)
                
                processing_stats['processed'] += 1
                processing_stats['total_chunks'] += len(chunks)
                processing_stats['total_chars'] += total_chars
                
                file_time = time.time() - file_start_time
                print(f"  ✓ Completed in {format_time(file_time)}")
                
                # Calculate ETA
                elapsed = time.time() - start_time
                files_remaining = len(all_files) - idx
                if idx > 0:
                    avg_time_per_file = elapsed / idx
                    eta = avg_time_per_file * files_remaining
                    print(f"  ⏱  Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:100]}")
                processing_stats['failed'] += 1
            
            # Update overall progress bar
            pbar_overall.update(1)
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total files: {processing_stats['total_files']}")
    print(f"  ✓ Processed: {processing_stats['processed']}")
    print(f"  ✗ Failed: {processing_stats['failed']}")
    print(f"\nChunk statistics:")
    print(f"  • Total chunks: {processing_stats['total_chunks']:,}")
    print(f"  • Total characters: {processing_stats['total_chars']:,}")
    if processing_stats['processed'] > 0:
        print(f"  • Average chunks per file: {processing_stats['total_chunks']//processing_stats['processed']}")
    print("="*80)
    
    if len(all_chunks) == 0:
        print("\n✗ No chunks generated.")
        return
    
    # Build vector database
    print("\n[5/5] Building vector database...")
    print(f"  Creating embeddings for {len(all_chunks):,} chunks...")
    
    db_start_time = time.time()
    
    from langchain_community.vectorstores import Chroma
    
    # Build with progress bar
    batch_size = 50
    vectordb = None
    
    with tqdm(total=len(all_chunks), desc="Creating embeddings", 
              unit="chunk", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_embed:
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            
            if vectordb is None:
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=db_builder.embeddings,
                    persist_directory=output_path
                )
            else:
                vectordb.add_documents(batch)
            
            pbar_embed.update(len(batch))
    
    vectordb.persist()
    
    db_time = time.time() - db_start_time
    print(f"\n  ✓ Database built in {format_time(db_time)}")
    print(f"  ✓ Saved to: {output_path}")
    
    # Test retrieval
    print("\n" + "="*80)
    print("TESTING RETRIEVAL")
    print("="*80)
    
    test_queries = [
        "近视手术后注意事项",
        "角膜炎的症状"
    ]
    
    for query in test_queries:
        print(f"\n✓ Query: '{query}'")
        results = vectordb.similarity_search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata.get('source', 'unknown')[:50]} (p.{doc.metadata.get('page', 'N/A')})")
            print(f"      {doc.page_content[:100]}...")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✓ BUILD COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Total time: {format_time(total_time)}")
    print(f"Documents: {processing_stats['processed']}/{processing_stats['total_files']}")
    print(f"Chunks: {processing_stats['total_chunks']:,}")
    print(f"Database: {output_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
