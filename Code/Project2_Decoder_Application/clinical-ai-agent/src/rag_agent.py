"""
Advanced RAG Agent with Semantic Chunking
Supports HuggingFace Inference API and smart chunking strategies
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import requests
import os

class SemanticTextChunker:
    """
    Advanced semantic chunking using embedding similarity
    Better than fixed-size chunking for maintaining context
    """
    
    def __init__(self, embeddings, breakpoint_threshold="percentile"):
        """
        Args:
            embeddings: HuggingFace embeddings model
            breakpoint_threshold: "percentile", "standard_deviation", or "interquartile"
        """
        self.semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold
        )
    
    def split_text(self, text: str, metadata: dict = None) -> List[Document]:
        """Split text using semantic similarity"""
        docs = self.semantic_chunker.create_documents([text])
        
        # Add metadata
        if metadata:
            for i, doc in enumerate(docs):
                doc.metadata.update(metadata)
                doc.metadata['chunk_id'] = i
        
        return docs

class HybridChunker:
    """
    Hybrid chunking strategy that handles different content types
    - Plain text: Semantic chunking
    - Tables: Keep as single chunks
    - Mixed content: Smart separation
    """
    
    def __init__(self, embeddings, 
                 chunk_size=800, 
                 chunk_overlap=100,
                 use_semantic=True):
        """
        Args:
            embeddings: Embedding model
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            use_semantic: Use semantic chunking for text
        """
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic
        
        # Fallback to recursive chunking
        self.recursive_chunker = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        if use_semantic:
            try:
                self.semantic_chunker = SemanticTextChunker(embeddings)
            except:
                print("Warning: Semantic chunking not available, using recursive")
                self.use_semantic = False
    
    def chunk_document(self, doc_data: Dict) -> List[Document]:
        """
        Intelligently chunk a document based on content type
        
        Args:
            doc_data: Dictionary with 'text_content', 'tables', 'images'
        
        Returns:
            List of Document objects
        """
        all_chunks = []
        
        # Process text content
        for text_item in doc_data.get('text_content', []):
            text = text_item['text']
            
            if len(text.strip()) < 50:  # Skip very short text
                continue
            
            metadata = {
                'source': doc_data.get('source', 'unknown'),
                'page': text_item.get('page', 1),
                'content_type': 'text'
            }
            
            # Use semantic chunking for longer text
            if self.use_semantic and len(text) > 500:
                chunks = self.semantic_chunker.split_text(text, metadata)
            else:
                # Fallback to recursive chunking
                chunk_texts = self.recursive_chunker.split_text(text)
                chunks = [
                    Document(
                        page_content=chunk_text,
                        metadata={**metadata, 'chunk_id': i}
                    )
                    for i, chunk_text in enumerate(chunk_texts)
                ]
            
            all_chunks.extend(chunks)
        
        # Process tables - keep each table as a single chunk
        for table_item in doc_data.get('tables', []):
            table_text = table_item['text_representation']
            
            metadata = {
                'source': doc_data.get('source', 'unknown'),
                'page': table_item.get('page', 1),
                'content_type': 'table',
                'rows': table_item.get('rows', 0),
                'columns': table_item.get('columns', 0)
            }
            
            all_chunks.append(Document(
                page_content=table_text,
                metadata=metadata
            ))
        
        # Add image context (placeholder for future image analysis)
        for img_item in doc_data.get('images', []):
            metadata = {
                'source': doc_data.get('source', 'unknown'),
                'page': img_item.get('page', 1),
                'content_type': 'image',
                'image_index': img_item.get('image_index', 0)
            }
            
            # Placeholder text for image
            image_text = f"[Image {img_item.get('image_index')} on page {img_item.get('page')}]"
            
            all_chunks.append(Document(
                page_content=image_text,
                metadata=metadata
            ))
        
        return all_chunks

class HuggingFaceLLM:
    """
    HuggingFace Inference API wrapper
    Free tier available, no local GPU needed
    """
    
    def __init__(self, 
                 model_id="Qwen/Qwen2.5-7B-Instruct",
                 api_token=None):
        """
        Args:
            model_id: HuggingFace model ID
            api_token: HuggingFace API token (get from hf.co/settings/tokens)
        """
        self.model_id = model_id
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        
        if not self.api_token:
            raise ValueError("HuggingFace API token required!")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def generate(self, prompt: str, max_tokens=512, temperature=0.7) -> str:
        """Generate response using HuggingFace API"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            return str(result)
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

class MedicalRAGAgent:
    """
    Medical RAG Agent with advanced chunking and retrieval
    """
    
    def __init__(self, 
                 vectordb,
                 llm,
                 top_k=5,
                 similarity_threshold=0.5):
        """
        Args:
            vectordb: Vector database
            llm: Language model (HuggingFaceLLM)
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.vectordb = vectordb
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        self.retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": top_k,
                "score_threshold": similarity_threshold
            }
        )
    
    def query(self, question: str) -> Dict:
        """
        Answer patient question using RAG
        
        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'confidence': float
            }
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.get_relevant_documents(question)
        
        if not relevant_docs:
            return {
                'answer': "Sorry, I couldn't find relevant information in the knowledge base.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            content_type = doc.metadata.get('content_type', 'text')
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            
            context_parts.append(
                f"[Source {i} - {content_type} from {source}, page {page}]\n{doc.page_content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are a professional medical assistant. Answer the patient's question based on the provided medical knowledge.

Medical Knowledge:
{context}

Patient Question: {question}

Important:
1. Only use information from the provided medical knowledge
2. If the information is not in the knowledge, clearly state that
3. Use professional but easy-to-understand language
4. Always remind patients to consult a doctor for medical advice

Answer:"""
        
        # Generate answer
        answer = self.llm.generate(prompt, max_tokens=512)
        
        # Prepare sources
        sources = [
            {
                'source': doc.metadata.get('source', 'unknown'),
                'page': doc.metadata.get('page', 'unknown'),
                'content_type': doc.metadata.get('content_type', 'text'),
                'excerpt': doc.page_content[:200]
            }
            for doc in relevant_docs
        ]
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': len(relevant_docs) / self.top_k
        }
