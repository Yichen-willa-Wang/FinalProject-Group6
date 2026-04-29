"""
Vector Database Builder Module
Build and manage vector database for RAG
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent))
from pdf_processor import PDFProcessor

class ChineseTextSplitter:
    """Split Chinese text into chunks"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Chinese separators
        self.separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            ""
        ]
    
    def split_text(self, text, metadata=None):
        """Split Chinese text into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata['chunk_id'] = i
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents

class VectorDBBuilder:
    """Build and manage vector database"""
    
    def __init__(self, model_name="BAAI/bge-base-zh-v1.5", db_type="chroma"):
        """
        Initialize vector database builder
        
        Args:
            model_name: Embedding model name
            db_type: Database type ("chroma" or "faiss")
        """
        print(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.db_type = db_type
        self.text_splitter = ChineseTextSplitter()
        self.pdf_processor = PDFProcessor()
    
    def build_from_pdfs(self, pdf_folder, output_path):
        """
        Build vector database from PDF folder
        
        Args:
            pdf_folder: Path to PDF folder
            output_path: Output path for vector database
        """
        all_documents = []
        
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            
            # Extract content
            content = self.pdf_processor.process_pdf(str(pdf_file))
            
            # Process text content
            for page_content in content:
                metadata = {
                    'source': pdf_file.name,
                    'page': page_content['page'],
                    'type': 'text'
                }
                
                # Split into chunks
                docs = self.text_splitter.split_text(
                    page_content['text'],
                    metadata=metadata
                )
                all_documents.extend(docs)
        
        print(f"\nTotal documents: {len(all_documents)}")
        
        # Build vector database
        if self.db_type == "chroma":
            print("Building ChromaDB vector database...")
            vectordb = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=output_path
            )
            vectordb.persist()
        
        elif self.db_type == "faiss":
            print("Building FAISS vector database...")
            vectordb = FAISS.from_documents(
                documents=all_documents,
                embedding=self.embeddings
            )
            vectordb.save_local(output_path)
        
        print(f"Vector database saved to: {output_path}")
        return vectordb
    
    def load_vectordb(self, db_path):
        """Load existing vector database"""
        if self.db_type == "chroma":
            vectordb = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )
        elif self.db_type == "faiss":
            vectordb = FAISS.load_local(
                db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        return vectordb


class HybridChunker:
    """
    Hybrid chunking strategy
    """
    
    def __init__(self, embeddings, chunk_size=800, chunk_overlap=100, use_semantic=False):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic
        
        self.recursive_chunker = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def chunk_document(self, doc_data):
        """Intelligently chunk a document"""
        all_chunks = []
        
        # Process text content
        for text_item in doc_data.get('text_content', []):
            text = text_item['text']
            
            if len(text.strip()) < 50:
                continue
            
            metadata = {
                'source': doc_data.get('source', 'unknown'),
                'page': text_item.get('page', 1),
                'content_type': 'text'
            }
            
            chunk_texts = self.recursive_chunker.split_text(text)
            chunks = [
                Document(
                    page_content=chunk_text,
                    metadata={**metadata, 'chunk_id': i}
                )
                for i, chunk_text in enumerate(chunk_texts)
            ]
            
            all_chunks.extend(chunks)
        
        # Process tables
        for table_item in doc_data.get('tables', []):
            table_text = table_item.get('text_representation', '')
            
            metadata = {
                'source': doc_data.get('source', 'unknown'),
                'page': table_item.get('page', 1),
                'content_type': 'table'
            }
            
            all_chunks.append(Document(
                page_content=table_text,
                metadata=metadata
            ))
        
        return all_chunks
