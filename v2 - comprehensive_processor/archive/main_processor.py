#!/usr/bin/env python3
"""
Main Document Processing Module

Orchestrates the complete document processing pipeline.
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging

# Import our custom modules
from config_manager import ConfigManager
from file_scanner import FileScanner
from pdf_converter import PDFConverter
from ai_extractor import AIExtractor
from csv_manager import CSVManager


class DocumentProcessor:
    """Main class that orchestrates the document processing pipeline."""
    
    def __init__(self, config_file: str = "config.ini"):
        """
        Initialize the document processor.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_general_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.file_scanner = FileScanner(
            self.config['source_directory'],
            self.config['output_directory'],
            self.config['ignore_hash_paths']
        )
        
        self.pdf_converter = PDFConverter(max_pages=10)  # Process more pages for completeness
        
        self.ai_extractor = AIExtractor(
            self.config['model_name'],
            self.config['ollama_url']
        )
        
        self.csv_manager = CSVManager(self.config['output_directory'])
        
        # Get document type configurations
        self.field_definitions = self.config_manager.get_document_field_definitions()
        self.search_criteria = self.config_manager.get_document_search_criteria()
        
        logging.info("Document processor initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['log_directory'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_level = logging.DEBUG if self.config['verbose'] else logging.INFO
        log_file = log_dir / 'document_processor.log'
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.log_file = str(log_file)
        logging.info(f"Logging configured: {log_file}")
    
    def run_complete_pipeline(self):
        """Execute the complete document processing pipeline."""
        print("=" * 80)
        print("MEDICAL DOCUMENT PROCESSING PIPELINE")
        print("=" * 80)
        print(f"Source Directory: {self.config['source_directory']}")
        print(f"Output Directory: {self.config['output_directory']}")
        print(f"AI Model: {self.config['model_name']}")
        print(f"Log File: {self.log_file}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Mirror directory structure
            print("\nSTEP 1: Mirroring directory structure...")
            dirs_created = self.file_scanner.mirror_directory_structure()
            print(f"✓ Created {dirs_created} directories")
            
            # Step 2: Find and organize PDF files by patient
            print("\nSTEP 2: Finding and organizing PDF files by patient...")
            patients_with_pdfs = self._organize_pdfs_by_patient()
            
            if not patients_with_pdfs:
                print("No PDF files found for any patients. Processing complete.")
                return
            
            total_patients = len(patients_with_pdfs)
            total_pdfs = sum(len(pdfs) for pdfs in patients_with_pdfs.values())
            print(f"✓ Found {total_pdfs} PDF files for {total_patients} patients")
            
            # Step 3: Create CSV files
            print("\nSTEP 3: Setting up output CSV files...")
            
            # Create CSV files for each document type
            for doc_type in self.field_definitions.keys():
                self.csv_manager.create_document_type_csv(doc_type, self.field_definitions[doc_type])
            
            # Create patient data CSV
            patient_csv = self.csv_manager.create_patient_csv()
            print(f"✓ Created output CSV files")
            
            # Step 4: Process each patient individually
            print(f"\nSTEP 4: Processing patients individually...")
            print("=" * 60)
            
            processed_patients = 0
            
            for patient_id, (patient_folder, pdf_files) in enumerate(patients_with_pdfs.items(), 1):
                try:
                    print(f"\nPROCESSING PATIENT {patient_id}/{total_patients}")
                    print(f"Patient: {Path(patient_folder).name}")
                    print(f"Documents: {len(pdf_files)} PDF files")
                    print("-" * 40)
                    
                    # Process this patient's documents
                    patient_extractions = self._process_patient_documents(
                        patient_folder, pdf_files, patient_id, total_patients
                    )
                    
                    # Aggregate and write patient data immediately
                    if patient_extractions:
                        print(f"    Aggregating data for {Path(patient_folder).name}...")
                        aggregated_data = self.csv_manager.aggregate_patient_data(
                            patient_folder, patient_extractions
                        )
                        self.csv_manager.add_patient_record(aggregated_data)
                        
                        # Show immediate results
                        successful_docs = sum(1 for e in patient_extractions if e.get('success'))
                        print(f"✓ COMPLETED: {Path(patient_folder).name}")
                        print(f"  - Processed {successful_docs}/{len(pdf_files)} documents successfully")
                        print(f"  - Patient record written to: {self.csv_manager.get_csv_file_path('patient_data')}")
                        print(f"  - Document CSVs in: {self.config['output_directory']}")
                        print(f"  - Progress: {patient_id}/{total_patients} patients ({patient_id/total_patients*100:.1f}%)")
                        
                        processed_patients += 1
                    else:
                        print(f"✗ No successful extractions for {Path(patient_folder).name}")
                    
                except Exception as e:
                    logging.error(f"Error processing patient {patient_folder}: {e}")
                    print(f"✗ ERROR processing {Path(patient_folder).name}: {e}")
                    continue
            
            # Processing complete
            elapsed_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("PROCESSING COMPLETE")
            print("=" * 80)
            print(f"Total processing time: {elapsed_time:.1f} seconds")
            print(f"Patients processed successfully: {processed_patients}/{total_patients}")
            print(f"Average time per patient: {elapsed_time/max(processed_patients,1):.1f} seconds")
            
            # Show output files
            print("\nOutput files created:")
            print(f"  - Patient data: {patient_csv}")
            for doc_type in self.field_definitions.keys():
                csv_file = self.csv_manager.get_csv_file_path(doc_type)
                if csv_file and os.path.exists(csv_file):
                    print(f"  - {doc_type}: {csv_file}")
            
            print(f"\nLog file: {self.log_file}")
            print("\nReady for machine learning model training!")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            print(f"\nError: {e}")
            sys.exit(1)
    
    def _organize_pdfs_by_patient(self) -> Dict[str, List[Path]]:
        """
        Organize PDF files by patient folder.
        
        Returns:
            Dictionary mapping patient folder paths to lists of PDF files
        """
        # First find all PDFs
        pdfs_by_type = self.file_scanner.find_pdf_files(self.search_criteria)
        
        # Organize by patient
        patients_with_pdfs = defaultdict(list)
        
        for doc_type, pdf_list in pdfs_by_type.items():
            for pdf_path in pdf_list:
                patient_folder = self._get_patient_folder_from_path(pdf_path)
                if patient_folder:
                    patients_with_pdfs[patient_folder].append(pdf_path)
        
        # Log summary by document type
        for doc_type, pdfs in pdfs_by_type.items():
            print(f"  - {doc_type}: {len(pdfs)} files")
        
        return dict(patients_with_pdfs)
    
    def _process_patient_documents(self, patient_folder: str, pdf_files: List[Path], 
                                   patient_num: int, total_patients: int) -> List[Dict]:
        """
        Process all documents for a single patient.
        
        Args:
            patient_folder: Path to patient folder
            pdf_files: List of PDF files for this patient
            patient_num: Current patient number
            total_patients: Total number of patients
            
        Returns:
            List of extraction results for this patient
        """
        patient_extractions = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                print(f"    Document {i}/{len(pdf_files)}: {pdf_path.name}")
                
                # Check if text file already exists
                rel_path = pdf_path.relative_to(Path(self.config['source_directory']))
                text_file = Path(self.config['output_directory']) / rel_path.with_suffix('.pdf.txt')
                
                # Convert PDF to text only if not already done
                if not text_file.exists():
                    # Ensure output directory exists
                    text_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Convert PDF to text
                    print(f"      Converting PDF to text...")
                    text_content = self.pdf_converter.convert_pdf_to_text(pdf_path)
                    
                    if text_content and len(text_content.strip()) > 50:
                        # Write text file with metadata
                        with open(text_file, 'w', encoding='utf-8') as f:
                            f.write(f"Source: {pdf_path}\n")
                            f.write(f"Converted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(text_content)
                        print(f"      ✓ Converted to text ({len(text_content)} chars)")
                    else:
                        print(f"      ✗ Failed to extract meaningful text")
                        continue
                else:
                    print(f"      Using existing text file")
                
                # Classify document type based on filename (single classification)
                doc_type = self._classify_pdf(pdf_path.name)
                
                # Read text content (single read)
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"      Extracting data as {doc_type}...")
                
                # Extract data using AI
                fields = self.field_definitions[doc_type]
                extraction_result = self.ai_extractor.extract_data(
                    content, doc_type, fields
                )
                
                # Add metadata
                extraction_result.update({
                    'absolute_file_path': str(pdf_path.resolve()),
                    'document_type': doc_type,
                    'processing_success': extraction_result['success'],
                    'response_time_seconds': f"{extraction_result['response_time']:.2f}",
                    'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Add extracted fields to result for CSV writing
                if extraction_result['success']:
                    extraction_result.update(extraction_result['extracted_data'])
                
                # Write to document type CSV immediately
                csv_file = self.csv_manager.get_csv_file_path(doc_type)
                if csv_file:
                    self.csv_manager.add_document_record(csv_file, extraction_result)
                
                # Save detailed extraction data
                data_file = text_file.with_suffix('.txt__data.txt')
                self.ai_extractor.save_extraction_data(extraction_result, str(data_file))
                
                # Add to patient extractions
                extraction_result['document_type'] = doc_type
                patient_extractions.append(extraction_result)
                
                # Show progress
                if extraction_result['success']:
                    found_count = sum(1 for v in extraction_result['extracted_data'].values() 
                                    if v != "NOT FOUND")
                    print(f"      ✓ {doc_type}: {found_count}/{len(fields)} fields extracted ({extraction_result['response_time']:.1f}s)")
                else:
                    print(f"      ✗ {doc_type}: extraction failed")
                
            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}")
                print(f"      ✗ Error processing {pdf_path.name}: {e}")
                continue
        
        return patient_extractions
    
    def _classify_pdf(self, filename: str) -> str:
        """
        Classify a PDF file by document type.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Document type string
        """
        filename_lower = filename.lower()
        
        # Check specific document types first
        for doc_type, search_terms in self.search_criteria.items():
            if doc_type == 'general_medical_document':
                continue
                
            if any(term.lower() in filename_lower for term in search_terms):
                return doc_type
        
        # Default to general medical document
        return 'general_medical_document'
    
    def _get_patient_folder_from_path(self, file_path: Path) -> str:
        """
        Extract patient folder path from a file path.
        
        Args:
            file_path: Path to a file
            
        Returns:
            Patient folder path or None
        """
        path_parts = file_path.parts
        
        # Look for pattern: Practice Name/Portfolio #X/Patient Name
        for i, part in enumerate(path_parts):
            if "Portfolio #" in part and i + 1 < len(path_parts):
                # The next part should be the patient name
                patient_folder_parts = path_parts[:i+2]
                return str(Path(*patient_folder_parts))
        
        return None


def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process medical documents for insurance lawsuit prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This application processes medical documents through a complete pipeline:

1. Mirrors source directory structure to output directory
2. Finds and classifies PDF files by document type
3. Converts PDFs to text using OCR
4. Extracts structured data using qwen2.5:14b AI model
5. Aggregates patient data for machine learning

Requirements:
- config.ini with source_directory and output_directory
- Tesseract OCR installed
- Ollama running with qwen2.5:14b model

Output:
- Mirrored directory structure with text files
- patient_data.csv with aggregated patient information
- Separate CSV files for each document type
- Detailed extraction data files
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
        # Create processor and run pipeline
        processor = DocumentProcessor(config_file=args.config)
        
        if args.verbose:
            processor.config['verbose'] = True
            processor._setup_logging()
        
        processor.run_complete_pipeline()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()