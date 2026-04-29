#!/usr/bin/env python3
"""只处理Word文档并添加到现有向量库"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_processor import DocumentProcessor
from vector_db_builder import VectorDBBuilder, HybridChunker
from langchain_community.vectorstores import Chroma
import yaml

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()
pdf_folder = config['paths']['pdf_dir']
output_path = config['paths']['vector_db_dir']

print("Processing Word documents only...")

doc_processor = DocumentProcessor()
db_builder = VectorDBBuilder(model_name="BAAI/bge-base-zh-v1.5", db_type="chroma")

chunker = HybridChunker(
    embeddings=db_builder.embeddings,
    chunk_size=800,
    chunk_overlap=100,
    use_semantic=False
)

# 只处理Word文档
pdf_folder_path = Path(pdf_folder)
word_files = list(pdf_folder_path.glob("*.docx")) + list(pdf_folder_path.glob("*.doc"))

print(f"Found {len(word_files)} Word documents")

all_chunks = []
for idx, doc_file in enumerate(word_files, 1):
    print(f"[{idx}/{len(word_files)}] {doc_file.name}")
    
    try:
        doc_data = doc_processor.process_document(str(doc_file))
        doc_data['source'] = doc_file.name
        
        chunks = chunker.chunk_document(doc_data)
        all_chunks.extend(chunks)
        print(f"  ✓ {len(chunks)} chunks")
        
    except Exception as e:
        print(f"  ✗ {str(e)[:50]}")

print(f"\nTotal new chunks: {len(all_chunks)}")

if len(all_chunks) > 0:
    print("Adding to existing vector database...")
    
    # 加载现有数据库
    vectordb = Chroma(
        persist_directory=output_path,
        embedding_function=db_builder.embeddings
    )
    
    # 添加新chunks
    vectordb.add_documents(all_chunks)
    vectordb.persist()
    
    print(f"✓ Added {len(all_chunks)} chunks to database")
else:
    print("No chunks to add")

print("\nDone!")
