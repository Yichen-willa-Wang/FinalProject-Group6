#!/usr/bin/env python3
"""Split large PDF files - 10 pages per chunk"""

import fitz
from pathlib import Path
import shutil

def split_pdf(input_path, output_folder, pages_per_chunk=10, overlap=1):
    """Split PDF into very small chunks"""
    
    print(f"\nProcessing: {Path(input_path).name}")
    
    doc = fitz.open(input_path)
    total_pages = len(doc)
    
    print(f"  Total pages: {total_pages}")
    
    if total_pages <= 30:
        print(f"  Skip: small file")
        doc.close()
        return
    
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    base_name = Path(input_path).stem
    chunk_num = 1
    current_page = 0
    
    while current_page < total_pages:
        end_page = min(current_page + pages_per_chunk, total_pages)
        
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=current_page, to_page=end_page-1)
        
        output_path = output_folder / f"{base_name}_p{chunk_num:03d}.pdf"
        chunk_doc.save(str(output_path))
        chunk_doc.close()
        
        size_kb = output_path.stat().st_size / 1024
        print(f"    Part {chunk_num}: pg {current_page+1}-{end_page} ({size_kb:.0f}KB)")
        
        current_page = end_page - overlap
        
        if end_page >= total_pages:
            break
            
        chunk_num += 1
    
    doc.close()
    print(f"  Done: {chunk_num} chunks")

def main():
    input_folder = Path("/home/ubuntu/data/raw/medical_pdfs")
    output_folder = Path("/home/ubuntu/data/raw/medical_pdfs_split")
    large_files_folder = Path("/home/ubuntu/data/raw/medical_pdfs_large")
    
    print("="*80)
    print("PDF SPLITTER - 10 PAGES PER CHUNK")
    print("="*80)
    
    output_folder.mkdir(exist_ok=True)
    large_files_folder.mkdir(exist_ok=True)
    
    pdf_files = list(input_folder.glob("*.pdf"))
    large_files = []
    
    for pdf_file in pdf_files:
        try:
            doc = fitz.open(str(pdf_file))
            page_count = len(doc)
            doc.close()
            
            if page_count > 30:
                size_mb = pdf_file.stat().st_size / (1024 * 1024)
                large_files.append((pdf_file, size_mb, page_count))
        except:
            continue
    
    print(f"\nFound {len(large_files)} large files (>30 pages)")
    
    for idx, (pdf_file, size_mb, pages) in enumerate(large_files, 1):
        print(f"\n[{idx}/{len(large_files)}]")
        
        try:
            split_pdf(
                input_path=str(pdf_file),
                output_folder=output_folder,
                pages_per_chunk=10,
                overlap=1
            )
            
            new_path = large_files_folder / pdf_file.name
            pdf_file.rename(new_path)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Copy small files
    print(f"\n{'='*80}")
    print("Copying small files...")
    
    small_pdfs = [f for f in input_folder.glob("*.pdf")]
    word_files = list(input_folder.glob("*.docx")) + list(input_folder.glob("*.doc"))
    
    for f in small_pdfs + word_files:
        try:
            shutil.copy(str(f), str(output_folder / f.name))
        except:
            pass
    
    total_pdfs = len(list(output_folder.glob("*.pdf")))
    total_word = len(list(output_folder.glob("*.docx"))) + len(list(output_folder.glob("*.doc")))
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"Total PDFs: {total_pdfs}")
    print(f"Total Word: {total_word}")
    print(f"TOTAL: {total_pdfs + total_word}")
    print("="*80)

if __name__ == "__main__":
    main()
