#!/usr/bin/env python3
"""
PDF Converter Module

Handles conversion of PDF files to text using both text extraction and OCR.
"""

import os
import io
import time
from pathlib import Path
from typing import Tuple, List
import logging

try:
    import PyPDF2
    import pytesseract
    import fitz  # PyMuPDF
    from PIL import Image
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install PyPDF2 pytesseract pymupdf pillow")
    raise


class PDFConverter:
    """Converts PDF files to text using text extraction and OCR."""
    
    def __init__(self, max_pages: int = 10, ocr_timeout: int = 45):  # Increased for thoroughness
        """
        Initialize PDF converter.
        
        Args:
            max_pages: Maximum number of pages to process per PDF (increased for completeness)
            ocr_timeout: Timeout in seconds for OCR operations
        """
        self.max_pages = max_pages
        self.ocr_timeout = ocr_timeout
        self._verify_dependencies()
    
    def _verify_dependencies(self):
        """Verify that required dependencies are available."""
        try:
            # Test Tesseract
            pytesseract.image_to_string(Image.new('RGB', (100, 100), color='white'))
            logging.info("Tesseract OCR is available")
        except Exception as e:
            logging.error(f"Tesseract OCR not available: {e}")
            raise RuntimeError("Tesseract OCR is required but not available")
    
    def convert_pdf_to_text(self, pdf_path: Path) -> str:
        """
        Convert PDF to text using best available method.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        logging.info(f"Converting PDF to text: {pdf_path}")
        
        # Try text extraction first (faster)
        text = self._extract_text_directly(pdf_path)
        
        # If text extraction is insufficient, use OCR
        if not self._is_text_sufficient(text):
            logging.info(f"Text extraction insufficient, using OCR: {pdf_path}")
            text = self._extract_text_with_ocr(pdf_path)
        
        return text
    
    def _extract_text_directly(self, pdf_path: Path) -> str:
        """
        Extract text directly from PDF using PyPDF2.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text or empty string if failed
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                # Process limited number of pages
                num_pages = min(len(pdf_reader.pages), self.max_pages)
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}")
                    except Exception as e:
                        logging.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                return '\n\n'.join(text_parts)
                
        except Exception as e:
            logging.warning(f"Direct text extraction failed for {pdf_path}: {e}")
            return ""
    
    def _extract_text_with_ocr(self, pdf_path: Path) -> str:
        """
        Extract text using OCR with PyMuPDF and Tesseract.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            OCR extracted text
        """
        try:
            text_parts = []
            
            with fitz.open(str(pdf_path)) as pdf_document:
                # Process limited number of pages
                num_pages = min(len(pdf_document), self.max_pages)
                
                for page_num in range(num_pages):
                    try:
                        logging.debug(f"OCR processing page {page_num + 1}/{num_pages} of {pdf_path.name}")
                        
                        page = pdf_document[page_num]
                        
                        # Convert page to image with good resolution for accuracy
                        mat = fitz.Matrix(1.5, 1.5)  # Good balance of quality and speed
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("ppm")
                        
                        # Process with OCR using balanced settings
                        with Image.open(io.BytesIO(img_data)) as image:
                            page_text = pytesseract.image_to_string(
                                image,
                                lang='eng',
                                timeout=self.ocr_timeout,
                                config='--psm 6'  # Assume uniform block of text - works well for medical docs
                            )
                            
                            if page_text and page_text.strip():
                                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}")
                        
                        # Clean up
                        pix = None
                        
                        # Continue processing - don't stop early
                        combined_text = '\n\n'.join(text_parts)
                        # Remove early stopping - process all pages up to max_pages
                            
                    except Exception as e:
                        logging.warning(f"OCR failed on page {page_num + 1} of {pdf_path}: {e}")
                        continue
                
                final_text = '\n\n'.join(text_parts)
                
                if num_pages < len(pdf_document):
                    final_text += f"\n\n[Processed {num_pages} of {len(pdf_document)} pages]"
                
                return final_text
                
        except Exception as e:
            logging.error(f"OCR completely failed for {pdf_path}: {e}")
            return f"OCR_FAILED: Unable to extract text from {pdf_path.name}"
    
    def _is_text_sufficient(self, text: str) -> bool:
        """
        Check if extracted text is sufficient for processing.
        
        Args:
            text: Extracted text
            
        Returns:
            True if text is sufficient, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # More lenient threshold - if we get some meaningful text, use it
        word_count = len(text.split())
        if word_count < 10:  # Reduced from 20 - even small amounts of text can be valuable
            return False
        
        # Check if text has meaningful content (not just whitespace/symbols)
        meaningful_chars = sum(1 for c in text if c.isalnum())
        if meaningful_chars < len(text) * 0.2:  # Reduced from 30% - medical docs can have lots of formatting
            return False
        
        return True
    
    def convert_multiple_pdfs(self, pdf_paths: List[Path], output_dir: Path, source_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Convert multiple PDF files to text files.
        
        Args:
            pdf_paths: List of PDF file paths to convert
            output_dir: Output directory for text files
            source_dir: Source directory for calculating relative paths
            
        Returns:
            List of tuples (original_pdf_path, output_text_path)
        """
        converted_files = []
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                logging.info(f"Converting {i}/{len(pdf_paths)}: {pdf_path.name}")
                
                # Calculate output path
                rel_path = pdf_path.relative_to(source_dir)
                output_file = output_dir / rel_path.with_suffix('.pdf.txt')
                
                # Skip if already converted
                if output_file.exists():
                    logging.info(f"Already exists, skipping: {output_file}")
                    converted_files.append((pdf_path, output_file))
                    continue
                
                # Ensure output directory exists
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert PDF to text
                start_time = time.time()
                text_content = self.convert_pdf_to_text(pdf_path)
                elapsed = time.time() - start_time
                
                if text_content and len(text_content.strip()) > 50:
                    # Write text file with metadata
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"Source: {pdf_path}\n")
                        f.write(f"Converted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(text_content)
                    
                    converted_files.append((pdf_path, output_file))
                    logging.info(f"Converted in {elapsed:.1f}s: {pdf_path} -> {output_file}")
                else:
                    logging.error(f"Failed to extract meaningful text from: {pdf_path}")
                    
            except Exception as e:
                logging.error(f"Error converting {pdf_path}: {e}")
                continue
        
        return converted_files