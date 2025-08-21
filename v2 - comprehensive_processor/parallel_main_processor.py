#!/usr/bin/env python3
"""
Parallel Document Processing Module

Processes multiple documents simultaneously to dramatically improve speed.
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging
import concurrent.futures
import threading
from queue import Queue

# Import our custom modules
from config_manager import ConfigManager
from file_scanner import FileScanner
from pdf_converter import PDFConverter
from ai_extractor import AIExtractor
from csv_manager import CSVManager


class ParallelDocumentProcessor:
    """Parallel document processor for much faster processing."""
    
    def __init__(self, config_file: str = "config.ini", max_workers: int = 4):
        """
        Initialize the parallel document processor.
        
        Args:
            config_file: Path to configuration file
            max_workers: Number of parallel workers (recommend 4-8)
        """
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_general_config()
        self.max_workers = max_workers
        
        # Setup logging
        self._setup_logging()
        
        # Initialize shared components
        self.file_scanner = FileScanner(
            self.config['source_directory'],
            self.config['output_directory'],
            self.config['ignore_hash_paths']
        )
        
        self.csv_manager = CSVManager(self.config['output_directory'])
        
        # Get document type configurations
        self.field_definitions = self.config_manager.get_document_field_definitions()
        self.search_criteria = self.config_manager.get_document_search_criteria()
        
        # Thread-safe queue for results
        self.results_queue = Queue()
        self.csv_lock = threading.Lock()
        
        logging.info(f"Parallel processor initialized with {max_workers} workers")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['log_directory'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_level = logging.DEBUG if self.config['verbose'] else logging.INFO
        log_file = log_dir / 'parallel_processor.log'
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.log_file = str(log_file)
    
    def process_single_document(self, pdf_path: Path, patient_folder: str) -> Dict:
        """
        Process a single document (for parallel execution).
        
        Args:
            pdf_path: Path to PDF file
            patient_folder: Patient folder path
            
        Returns:
            Processing result dictionary
        """
        thread_name = threading.current_thread().name
        
        try:
            # Create thread-local instances to avoid conflicts
            pdf_converter = PDFConverter(max_pages=2, ocr_timeout=20)
            ai_extractor = AIExtractor(self.config['model_name'], self.config['ollama_url'])
            
            start_time = time.time()
            
            # Check if text file already exists
            rel_path = pdf_path.relative_to(Path(self.config['source_directory']))
            text_file = Path(self.config['output_directory']) / rel_path.with_suffix('.pdf.txt')
            
            # Convert PDF to text only if not already done
            if not text_file.exists():
                text_file.parent.mkdir(parents=True, exist_ok=True)
                
                logging.info(f"{thread_name}: Converting {pdf_path.name}")
                text_content = pdf_converter.convert_pdf_to_text(pdf_path)
                
                if text_content and len(text_content.strip()) > 50:
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(f"Source: {pdf_path}\n")
                        f.write(f"Converted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(text_content)
                else:
                    return {
                        'success': False,
                        'pdf_path': pdf_path,
                        'patient_folder': patient_folder,
                        'error': 'Failed to extract text',
                        'processing_time': time.time() - start_time
                    }
            
            # Classify and extract
            doc_type = self._classify_pdf(pdf_path.name)
            
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logging.info(f"{thread_name}: Extracting from {pdf_path.name} as {doc_type}")
            
            fields = self.field_definitions[doc_type]
            extraction_result = ai_extractor.extract_data(content, doc_type, fields)
            
            # Add metadata
            extraction_result.update({
                'pdf_path': pdf_path,
                'patient_folder': patient_folder,
                'absolute_file_path': str(pdf_path.resolve()),
                'document_type': doc_type,
                'processing_success': extraction_result['success'],
                'response_time_seconds': f"{extraction_result['response_time']:.2f}",
                'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': time.time() - start_time
            })
            
            if extraction_result['success']:
                extraction_result.update(extraction_result['extracted_data'])
            
            # Save detailed extraction data
            data_file = text_file.with_suffix('.txt__data.txt')
            ai_extractor.save_extraction_data(extraction_result, str(data_file))
            
            logging.info(f"{thread_name}: Completed {pdf_path.name} in {time.time() - start_time:.1f}s")
            return extraction_result
            
        except Exception as e:
            logging.error(f"{thread_name}: Error processing {pdf_path}: {e}")
            return {
                'success': False,
                'pdf_path': pdf_path,
                'patient_folder': patient_folder,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _classify_pdf(self, filename: str) -> str:
        """Classify PDF by document type."""
        filename_lower = filename.lower()
        
        for doc_type, search_terms in self.search_criteria.items():
            if doc_type == 'general_medical_document':
                continue
            if any(term.lower() in filename_lower for term in search_terms):
                return doc_type
        
        return 'general_medical_document'
    
    def process_patient_parallel(self, patient_folder: str, pdf_files: List[Path]) -> List[Dict]:
        """
        Process all documents for a patient in parallel.
        
        Args:
            patient_folder: Patient folder path
            pdf_files: List of PDF files
            
        Returns:
            List of processing results
        """
        print(f"  Processing {len(pdf_files)} documents in parallel...")
        
        # Process documents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents for processing
            future_to_pdf = {
                executor.submit(self.process_single_document, pdf_path, patient_folder): pdf_path 
                for pdf_path in pdf_files
            }
            
            results = []
            completed = 0
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if result['success']:
                        found_count = sum(1 for v in result['extracted_data'].values() 
                                        if v != "NOT FOUND") if 'extracted_data' in result else 0
                        print(f"    ✓ {pdf_path.name}: {found_count} fields ({result['processing_time']:.1f}s)")
                    else:
                        print(f"    ✗ {pdf_path.name}: {result.get('error', 'Failed')}")
                    
                    print(f"    Progress: {completed}/{len(pdf_files)} documents")
                    
                except Exception as e:
                    logging.error(f"Error getting result for {pdf_path}: {e}")
                    results.append({
                        'success': False,
                        'pdf_path': pdf_path,
                        'patient_folder': patient_folder,
                        'error': str(e)
                    })
        
        return results
    
    def run_parallel_pipeline(self):
        """Execute the parallel document processing pipeline."""
        print("=" * 80)
        print("PARALLEL MEDICAL DOCUMENT PROCESSING PIPELINE")
        print("=" * 80)
        print(f"Source Directory: {self.config['source_directory']}")
        print(f"Output Directory: {self.config['output_directory']}")
        print(f"AI Model: {self.config['model_name']}")
        print(f"Parallel Workers: {self.max_workers}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Setup
            print("\nSETUP: Preparing directories and CSV files...")
            self.file_scanner.mirror_directory_structure()
            
            # Create CSV files
            for doc_type in self.field_definitions.keys():
                self.csv_manager.create_document_type_csv(doc_type, self.field_definitions[doc_type])
            patient_csv = self.csv_manager.create_patient_csv()
            
            # Organize PDFs by patient
            print("\nFINDING: Organizing PDF files by patient...")
            patients_with_pdfs = self._organize_pdfs_by_patient()
            
            if not patients_with_pdfs:
                print("No PDF files found. Processing complete.")
                return
            
            total_patients = len(patients_with_pdfs)
            total_pdfs = sum(len(pdfs) for pdfs in patients_with_pdfs.values())
            print(f"✓ Found {total_pdfs} PDF files for {total_patients} patients")
            
            # Process patients
            print(f"\nPROCESSING: {total_patients} patients with {self.max_workers} parallel workers...")
            print("=" * 60)
            
            processed_patients = 0
            
            for patient_id, (patient_folder, pdf_files) in enumerate(patients_with_pdfs.items(), 1):
                try:
                    patient_name = Path(patient_folder).name
                    print(f"\nPATIENT {patient_id}/{total_patients}: {patient_name}")
                    print(f"Documents: {len(pdf_files)} PDF files")
                    print("-" * 40)
                    
                    # Process patient documents in parallel
                    patient_start = time.time()
                    patient_extractions = self.process_patient_parallel(patient_folder, pdf_files)
                    patient_time = time.time() - patient_start
                    
                    # Write to document CSVs (thread-safe)
                    with self.csv_lock:
                        for result in patient_extractions:
                            if result.get('success'):
                                doc_type = result.get('document_type', 'general_medical_document')
                                csv_file = self.csv_manager.get_csv_file_path(doc_type)
                                if csv_file:
                                    self.csv_manager.add_document_record(csv_file, result)
                    
                    # Aggregate and write patient data
                    if patient_extractions:
                        aggregated_data = self.csv_manager.aggregate_patient_data(
                            patient_folder, patient_extractions
                        )
                        
                        with self.csv_lock:
                            self.csv_manager.add_patient_record(aggregated_data)
                        
                        successful_docs = sum(1 for e in patient_extractions if e.get('success'))
                        print(f"\n✓ COMPLETED: {patient_name}")
                        print(f"  - Processed {successful_docs}/{len(pdf_files)} documents successfully")
                        print(f"  - Total time: {patient_time:.1f} seconds")
                        print(f"  - Average per document: {patient_time/len(pdf_files):.1f} seconds")
                        print(f"  - Speedup vs sequential: ~{self.max_workers}x faster")
                        print(f"  - Progress: {patient_id}/{total_patients} patients ({patient_id/total_patients*100:.1f}%)")
                        
                        processed_patients += 1
                    
                except Exception as e:
                    logging.error(f"Error processing patient {patient_folder}: {e}")
                    print(f"✗ ERROR processing {Path(patient_folder).name}: {e}")
                    continue
            
            # Final summary
            elapsed_time = time.time() - start_time
            
            print("\n" + "=" * 80)
            print("PARALLEL PROCESSING COMPLETE")
            print("=" * 80)
            print(f"Total processing time: {elapsed_time/60:.1f} minutes")
            print(f"Patients processed successfully: {processed_patients}/{total_patients}")
            print(f"Average time per patient: {elapsed_time/max(processed_patients,1)/60:.1f} minutes")
            print(f"Estimated speedup: ~{self.max_workers}x faster than sequential")
            print(f"Patient data written to: {patient_csv}")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            print(f"\nError: {e}")
            sys.exit(1)
    
    def _organize_pdfs_by_patient(self) -> Dict[str, List[Path]]:
        """Organize PDF files by patient folder."""
        pdfs_by_type = self.file_scanner.find_pdf_files(self.search_criteria)
        patients_with_pdfs = defaultdict(list)
        
        for doc_type, pdf_list in pdfs_by_type.items():
            for pdf_path in pdf_list:
                patient_folder = self._get_patient_folder_from_path(pdf_path)
                if patient_folder:
                    patients_with_pdfs[patient_folder].append(pdf_path)
        
        return dict(patients_with_pdfs)
    
    def _get_patient_folder_from_path(self, file_path: Path) -> str:
        """Extract patient folder path from a file path."""
        path_parts = file_path.parts
        
        for i, part in enumerate(path_parts):
            if "Portfolio #" in part and i + 1 < len(path_parts):
                patient_folder_parts = path_parts[:i+2]
                return str(Path(*patient_folder_parts))
        
        return None


def main():
    """Main entry point for parallel processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process medical documents in parallel for much faster processing"
    )
    
    parser.add_argument('--config', default='config.ini', help='Configuration file path')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        processor = ParallelDocumentProcessor(
            config_file=args.config, 
            max_workers=args.workers
        )
        
        if args.verbose:
            processor.config['verbose'] = True
            processor._setup_logging()
        
        processor.run_parallel_pipeline()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()