#!/usr/bin/env python3
"""
Combined Bill Processing System

This program processes medical bills through the complete pipeline:
1. Mirrors source directory structure to output directory
2. Converts PDF bills to text using OCR
3. Extracts structured data using AI
4. Outputs results to CSV

Required packages:
pip install PyPDF2 pytesseract pymupdf pillow requests configparser

System requirements:
- Tesseract OCR installed
- Ollama with qwen2.5:14b model
"""

import os
import sys
import shutil
import argparse
import configparser
import io
import csv
import json
import requests
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

try:
    import PyPDF2
    import pytesseract
    import fitz  # PyMuPDF
    from PIL import Image
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install PyPDF2 pytesseract pymupdf pillow requests")
    sys.exit(1)


class BillProcessor:
    def __init__(self, config_file: str = "config.ini"):
        """Initialize the bill processor with configuration."""
        self.config_file = config_file
        self.config = self.load_config()
        self.model_name = "qwen2.5:14b"
        self.ollama_url = "http://localhost:11434"
        self.setup_logging()
        self.check_model_available()
        self.bill_results = []  # Store all extracted bill data
    
    def load_config(self) -> Dict:
        """Load configuration from config.ini file."""
        config = configparser.ConfigParser()
        
        if not os.path.exists(self.config_file):
            print(f"Error: Config file '{self.config_file}' not found.")
            sys.exit(1)
        
        config.read(self.config_file)
        
        if 'General' not in config:
            print("Error: Config file must contain a [General] section")
            sys.exit(1)
        
        general = config['General']
        source_dir = general.get('source_directory', '').strip()
        output_dir = general.get('output_directory', '').strip()
        
        if not source_dir or not output_dir:
            print("Error: source_directory and output_directory must be specified in config.ini")
            sys.exit(1)
        
        # Get medical_bill fields from FileTypes section
        medical_bill_fields = []
        if 'FileTypes' in config and 'medical_bill' in config['FileTypes']:
            fields_str = config['FileTypes']['medical_bill']
            medical_bill_fields = [field.strip() for field in fields_str.split('`')]
        else:
            print("Error: [FileTypes] section with medical_bill entry required in config.ini")
            sys.exit(1)
        
        # Get ignore_hash_paths setting
        ignore_hash_paths = general.getboolean('ignore_hash_paths', fallback=False)
        
        return {
            'source_directory': source_dir,
            'output_directory': output_dir,
            'log_directory': general.get('log_directory', './logs').strip(),
            'verbose': general.getboolean('verbose', fallback=False),
            'ignore_hash_paths': ignore_hash_paths,
            'medical_bill_fields': medical_bill_fields
        }
    
    def setup_logging(self):
        """Set up logging configuration."""
        level = logging.DEBUG if self.config['verbose'] else logging.INFO
        
        log_dir = Path(self.config['log_directory'])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'bill_processor.log'
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.log_file = str(log_file)
    
    def check_model_available(self):
        """Check if qwen2.5:14b model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                if not any('qwen2.5:14b' in model for model in available_models):
                    print("Error: qwen2.5:14b model not found.")
                    print("Please install with: ollama pull qwen2.5:14b")
                    print("Available models:", available_models)
                    sys.exit(1)
                else:
                    print("✓ qwen2.5:14b model available")
            else:
                print("Error: Ollama not responding. Please start with: ollama serve")
                sys.exit(1)
        except Exception as e:
            print(f"Error: Cannot connect to Ollama: {e}")
            print("Please ensure Ollama is running: ollama serve")
            sys.exit(1)
    
    def mirror_directory_structure(self) -> int:
        """Mirror source directory structure to output directory."""
        print("STEP 1: Mirroring directory structure...")
        
        source_path = Path(self.config['source_directory'])
        output_path = Path(self.config['output_directory'])
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_path}")
        
        # Create output directory structure
        directories_created = 0
        
        for root, dirs, files in os.walk(source_path):
            # Calculate relative path from source
            rel_path = Path(root).relative_to(source_path)
            dest_dir = output_path / rel_path
            
            # Create directory if it doesn't exist
            if not dest_dir.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                directories_created += 1
                logging.debug(f"Created directory: {dest_dir}")
        
        print(f"✓ Created {directories_created} directories in output structure")
        return directories_created
    
    def find_bill_pdfs(self) -> List[Path]:
        """Find all PDF files containing 'bill' in the filename."""
        print("STEP 2: Finding PDF bills...")
        
        source_path = Path(self.config['source_directory'])
        bill_pdfs = []
        ignored_count = 0
        
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if (file.lower().endswith('.pdf') and 
                    'bill' in file.lower()):
                    bill_path = Path(root) / file
                    
                    # Check if we should ignore paths with directories starting with '#'
                    if self.config['ignore_hash_paths']:
                        # Check if any directory in the path starts with '#'
                        path_parts = bill_path.parts
                        has_hash_start = any(part.startswith('#') for part in path_parts[:-1])  # Exclude filename itself
                        
                        if has_hash_start:
                            ignored_count += 1
                            logging.info(f"Ignored bill PDF (hash directory): {bill_path}")
                            continue
                    
                    bill_pdfs.append(bill_path)
                    logging.info(f"Found bill PDF: {bill_path}")
        
        if self.config['ignore_hash_paths']:
            print(f"✓ Found {len(bill_pdfs)} bill PDFs (ignored {ignored_count} with directories starting with '#')")
        else:
            print(f"✓ Found {len(bill_pdfs)} bill PDFs")
        
        return bill_pdfs
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyPDF2 first, then OCR if needed."""
        # Try text extraction first
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(min(len(pdf_reader.pages), 5)):  # Limit to 5 pages
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # Check if extraction was meaningful
                if text.strip() and len(text.split()) >= 20:
                    logging.info(f"Using text extraction for {pdf_path}")
                    return text.strip()
                else:
                    logging.info(f"Text extraction insufficient for {pdf_path}, trying OCR")
        except Exception as e:
            logging.warning(f"Text extraction failed for {pdf_path}: {e}")
        
        # Fall back to OCR with better error handling
        return self.ocr_pdf_to_text_robust(pdf_path)
    
    def ocr_pdf_to_text_robust(self, pdf_path: Path) -> str:
        """Convert PDF to text using OCR with robust error handling."""
        try:
            logging.info(f"Using OCR for {pdf_path}")
            
            # Open document with context manager for proper cleanup
            with fitz.open(str(pdf_path)) as pdf_document:
                text_parts = []
                
                # Process maximum 3 pages for efficiency
                max_pages = min(len(pdf_document), 3)
                
                for page_num in range(max_pages):
                    try:
                        logging.debug(f"OCR processing page {page_num+1}/{max_pages} of {pdf_path.name}")
                        
                        # Get the page
                        page = pdf_document[page_num]
                        
                        # Create pixmap with reasonable resolution
                        mat = fitz.Matrix(1.2, 1.2)  # Lower resolution for speed
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        
                        # Process with PIL
                        with Image.open(io.BytesIO(img_data)) as image:
                            # Use OCR with timeout
                            try:
                                page_text = pytesseract.image_to_string(
                                    image, 
                                    lang='eng',
                                    timeout=20,
                                    config='--psm 6'  # Assume uniform block of text
                                )
                                
                                if page_text.strip():
                                    text_parts.append(f"--- Page {page_num+1} ---\n{page_text.strip()}")
                                    
                                    # Stop if we have enough text
                                    combined_text = '\n\n'.join(text_parts)
                                    if len(combined_text) > 3000:
                                        logging.info(f"Sufficient text extracted from {pdf_path.name}")
                                        break
                                        
                            except pytesseract.TesseractError as ocr_error:
                                logging.warning(f"OCR failed on page {page_num+1} of {pdf_path}: {ocr_error}")
                                continue
                                
                        # Clean up pixmap
                        pix = None
                        
                    except Exception as page_error:
                        logging.warning(f"Failed to process page {page_num+1} of {pdf_path}: {page_error}")
                        continue
                
                # Combine all text parts
                final_text = '\n\n'.join(text_parts)
                
                if max_pages < len(pdf_document):
                    final_text += f"\n\n[Processed {max_pages} of {len(pdf_document)} pages]"
                
                return final_text.strip()
                
        except Exception as e:
            logging.error(f"OCR completely failed for {pdf_path}: {e}")
            # Return a minimal text to avoid complete failure
            return f"OCR_FAILED: Unable to extract text from {pdf_path.name}"
    
    def convert_bills_to_text(self, bill_pdfs: List[Path]) -> List[Tuple[Path, Path]]:
        """Convert all bill PDFs to text files in output directory."""
        print("STEP 2: Converting PDF bills to text...")
        
        source_path = Path(self.config['source_directory'])
        output_path = Path(self.config['output_directory'])
        converted_bills = []
        
        total_bills = len(bill_pdfs)
        
        for i, pdf_path in enumerate(bill_pdfs, 1):
            try:
                print(f"Converting {i}/{total_bills}: {pdf_path.name}")
                
                # Calculate relative path and output location
                rel_path = pdf_path.relative_to(source_path)
                output_file = output_path / rel_path.with_suffix('.pdf.txt')
                
                # Skip if already converted (for resuming)
                if output_file.exists():
                    logging.info(f"Already exists, skipping: {output_file}")
                    converted_bills.append((pdf_path, output_file))
                    continue
                
                # Extract text from PDF with timeout protection
                start_time = time.time()
                text_content = self.extract_text_from_pdf(pdf_path)
                elapsed = time.time() - start_time
                
                if text_content and len(text_content.strip()) > 50:  # Ensure we have meaningful content
                    # Write text file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"Source: {pdf_path}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(text_content)
                    
                    converted_bills.append((pdf_path, output_file))
                    logging.info(f"Converted in {elapsed:.1f}s: {pdf_path} -> {output_file}")
                    print(f"  ✓ Converted in {elapsed:.1f}s ({len(text_content)} chars)")
                else:
                    logging.error(f"Failed to extract meaningful text from: {pdf_path}")
                    print(f"  ✗ No meaningful text extracted")
                    
            except Exception as e:
                logging.error(f"Error converting {pdf_path}: {e}")
                print(f"  ✗ Error: {e}")
        
        print(f"✓ Converted {len(converted_bills)} bills to text")
        return converted_bills
    
    def query_model(self, prompt: str) -> str:
        """Query the qwen2.5:14b model."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 300,
                "num_ctx": 4096,
                "top_k": 10,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            raise Exception(f"Error querying model: {e}")
    
    def create_extraction_prompt(self, content: str, fields: List[str]) -> str:
        """Create prompt for extracting bill information."""
        fields_list = '\n'.join([f"- {field}" for field in fields])
        
        return f"""TASK: Extract medical billing information from this document.

DOCUMENT:
{content}

EXTRACT THESE FIELDS:
{fields_list}

INSTRUCTIONS:
- For each field, find the exact value from the document
- If a field is not found, write "NOT FOUND"
- Return only the raw values, no explanations or context
- Format: "Field Name: Value" (one per line)
- Be precise - extract only what is requested

EXTRACT NOW:"""
    
    def parse_extraction_response(self, response: str, expected_fields: List[str]) -> Dict[str, str]:
        """Parse the model's extraction response."""
        extracted = {}
        
        # Initialize all fields as "NOT FOUND"
        for field in expected_fields:
            extracted[field] = "NOT FOUND"
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field_name = parts[0].strip()
                    field_value = parts[1].strip()
                    
                    # Find matching field (case insensitive, partial match)
                    for expected_field in expected_fields:
                        if (field_name.lower() in expected_field.lower() or 
                            expected_field.lower() in field_name.lower()):
                            if field_value and field_value.upper() not in ['NOT FOUND', 'N/A', 'NULL', 'NONE']:
                                extracted[expected_field] = field_value
                            break
        
        return extracted
    
    def write_csv_record(self, result: Dict, is_first_record: bool = False):
        """Write a single result to CSV immediately."""
        fields = self.config['medical_bill_fields']
        output_file = "output.csv"
        
        # CSV headers as specified
        headers = [
            'absolute file path',
            'file_type', 
            'model_used'
        ] + fields
        
        # Determine if we need to write headers
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers only for the very first record
            if is_first_record and not file_exists:
                writer.writerow(headers)
                print(f"Created {output_file} with headers")
            
            # Write data row
            if result.get('success'):
                row = [
                    result['original_pdf'],
                    result['file_type'],
                    result['model_used']
                ]
                
                # Add field values in order
                for field in fields:
                    value = result['extracted_data'].get(field, 'NOT FOUND')
                    row.append(value)
                
                writer.writerow(row)
                print(f"  → Added record to output.csv")
            else:
                # Write error row
                error_row = [
                    result.get('original_pdf', ''),
                    'ERROR',
                    result.get('error', 'Unknown error')
                ]
                error_row.extend([''] * len(fields))
                writer.writerow(error_row)
                print(f"  → Added error record to output.csv")
    
    def process_all_bills(self):
        """Run the complete bill processing pipeline."""
        print("="*70)
        print("BILL PROCESSING PIPELINE")
        print("="*70)
        print(f"Source: {self.config['source_directory']}")
        print(f"Output: {self.config['output_directory']}")
        print(f"Model: {self.model_name}")
        print(f"Log: {self.log_file}")
        print("="*70)
        
        try:
            # Step 1: Mirror directory structure
            self.mirror_directory_structure()
            
            # Step 2a: Find bill PDFs
            bill_pdfs = self.find_bill_pdfs()
            if not bill_pdfs:
                print("No bill PDFs found. Processing complete.")
                return
            
            # Step 2b: Convert PDFs to text
            text_files = self.convert_bills_to_text(bill_pdfs)
            if not text_files:
                print("No bills were successfully converted. Processing complete.")
                return
            
            # Step 3: Extract data using AI (with immediate CSV writing)
            if not text_files:
                print("No text files to process.")
                return
                
            print("STEP 3: Extracting bill data using qwen2.5:14b...")
            
            fields = self.config['medical_bill_fields']
            all_results = []
            
            total_files = len(text_files)
            first_record = True
            
            for i, (original_pdf, text_file) in enumerate(text_files, 1):
                try:
                    print(f"Processing {i}/{total_files}: {text_file.name}")
                    
                    # Read text file
                    with open(text_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Limit content size for model processing
                    if len(content) > 8000:
                        content = content[:8000] + "\n[Content truncated...]"
                    
                    # Create extraction prompt
                    prompt = self.create_extraction_prompt(content, fields)
                    
                    # Query model
                    start_time = time.time()
                    ai_response = self.query_model(prompt)
                    elapsed = time.time() - start_time
                    
                    # Parse response
                    extracted_data = self.parse_extraction_response(ai_response, fields)
                    
                    # Store result
                    result = {
                        'success': True,
                        'original_pdf': str(original_pdf.resolve()),  # Absolute path
                        'text_file': str(text_file),
                        'file_type': 'medical_bill',
                        'model_used': self.model_name,
                        'response_time': f"{elapsed:.1f}s",
                        'extracted_data': extracted_data,
                        'raw_response': ai_response
                    }
                    
                    # Write to CSV immediately
                    self.write_csv_record(result, is_first_record=first_record)
                    first_record = False
                    
                    all_results.append(result)
                    
                    # Show progress
                    found_count = sum(1 for v in extracted_data.values() if v != "NOT FOUND")
                    print(f"  ✓ Extracted {found_count}/{len(fields)} fields ({elapsed:.1f}s)")
                    
                    logging.info(f"Processed {text_file}: {found_count}/{len(fields)} fields")
                    
                except Exception as e:
                    error_result = {
                        'success': False,
                        'original_pdf': str(original_pdf.resolve()),
                        'text_file': str(text_file),
                        'error': str(e)
                    }
                    
                    # Write error to CSV immediately
                    self.write_csv_record(error_result, is_first_record=first_record)
                    first_record = False
                    
                    all_results.append(error_result)
                    print(f"  ✗ Error: {e}")
                    logging.error(f"Error processing {text_file}: {e}")
            
            print(f"\n✓ Processed {len(all_results)} bill files total")
            results = all_results
            
            # Summary
            successful = sum(1 for r in results if r.get('success'))
            print("="*70)
            print("PROCESSING COMPLETE")
            print("="*70)
            print(f"Total bills found: {len(bill_pdfs)}")
            print(f"Successfully converted: {len(text_files)}")
            print(f"Successfully processed: {successful}")
            print(f"Output file: output.csv (written incrementally)")
            print(f"Log file: {self.log_file}")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            print(f"Error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process medical bills from PDF to structured CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This program processes medical bills through a complete pipeline:

1. Mirrors source directory structure to output directory
2. Finds PDF files containing 'bill' in filename
3. Converts PDFs to text using OCR
4. Extracts structured data using qwen2.5:14b AI model
5. Outputs results to CSV file

Requirements:
- config.ini with [General] and [FileTypes] sections
- Tesseract OCR installed
- Ollama running with qwen2.5:14b model
- PDF files with 'bill' in filename

Output:
- Mirrored directory structure in output directory
- Text files (.pdf.txt) for each bill
- output.csv with extracted structured data
        """
    )
    
    parser.add_argument(
        '--config', 
        default='config.ini',
        help='Configuration file path (default: config.ini)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        processor = BillProcessor(config_file=args.config)
        if args.verbose:
            processor.config['verbose'] = True
            processor.setup_logging()
        
        processor.process_all_bills()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()