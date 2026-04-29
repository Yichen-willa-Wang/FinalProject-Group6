"""
PDF Processing Module
Handles both scanned and normal PDFs with OCR support
"""

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict

class PDFProcessor:
    """Process PDF documents and extract text"""
    
    def __init__(self):
        self.tesseract_config = '--psm 6 --oem 3'
    
    def is_scanned_pdf(self, pdf_path, sample_pages=3):
        """
        Check if PDF is scanned (image-based)
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to check
            
        Returns:
            True if PDF is scanned, False otherwise
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_check = min(sample_pages, total_pages)
        
        text_ratio = 0
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text()
            if len(text.strip()) < 100:
                text_ratio += 1
        
        doc.close()
        return text_ratio / pages_to_check > 0.5
    
    def extract_text_from_normal_pdf(self, pdf_path):
        """Extract text from normal PDF"""
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_content.append({
                'page': page_num + 1,
                'text': text,
                'type': 'text'
            })
        
        doc.close()
        return text_content
    
    def extract_text_from_scanned_pdf(self, pdf_path):
        """Extract text from scanned PDF using OCR"""
        images = convert_from_path(pdf_path, dpi=300)
        text_content = []
        
        for page_num, image in enumerate(images):
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            _, binary = cv2.threshold(
                gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            text = pytesseract.image_to_string(
                binary,
                lang='chi_sim+chi_tra',
                config=self.tesseract_config
            )
            
            text_content.append({
                'page': page_num + 1,
                'text': text,
                'type': 'ocr'
            })
        
        return text_content
    
    def process_pdf(self, pdf_path):
        """Smart PDF processing - auto-detect and process"""
        print(f"Processing PDF: {pdf_path}")
        
        if self.is_scanned_pdf(pdf_path):
            print("Detected scanned PDF, using OCR...")
            return self.extract_text_from_scanned_pdf(pdf_path)
        else:
            print("Detected normal PDF, extracting text...")
            return self.extract_text_from_normal_pdf(pdf_path)
