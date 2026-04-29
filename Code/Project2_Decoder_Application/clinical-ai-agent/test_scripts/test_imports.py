"""
Test script to verify all modules can be imported
"""

import sys
from pathlib import Path
sys.path.append('src')

print("Testing imports...")

try:
    from document_processor import DocumentProcessor
    print("✓ document_processor")
except Exception as e:
    print(f"✗ document_processor: {e}")

try:
    from pdf_processor import PDFProcessor
    print("✓ pdf_processor")
except Exception as e:
    print(f"✗ pdf_processor: {e}")

try:
    from vector_db_builder import VectorDBBuilder
    print("✓ vector_db_builder")
except Exception as e:
    print(f"✗ vector_db_builder: {e}")

try:
    from rag_agent import HuggingFaceLLM, MedicalRAGAgent
    print("✓ rag_agent")
except Exception as e:
    print(f"✗ rag_agent: {e}")

try:
    from app_gradio import MedicalChatBot
    print("✓ app_gradio")
except Exception as e:
    print(f"✗ app_gradio: {e}")

print("\nAll imports successful!")
