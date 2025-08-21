#!/usr/bin/env python3
"""
File Scanner Module

Handles directory structure mirroring and PDF file discovery.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set
import logging


class FileScanner:
    """Scans source directory for PDF files and manages directory structure."""
    
    def __init__(self, source_dir: str, output_dir: str, ignore_hash_paths: bool = True):
        """Initialize file scanner with source and output directories."""
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.ignore_hash_paths = ignore_hash_paths
        
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")
    
    def mirror_directory_structure(self) -> int:
        """
        Mirror source directory structure to output directory.
        
        Returns:
            Number of directories created
        """
        logging.info("Mirroring directory structure...")
        
        directories_created = 0
        
        # Walk through source directory structure
        for root, dirs, files in os.walk(self.source_dir):
            # Calculate relative path from source
            rel_path = Path(root).relative_to(self.source_dir)
            dest_dir = self.output_dir / rel_path
            
            # Create directory if it doesn't exist
            if not dest_dir.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                directories_created += 1
                logging.debug(f"Created directory: {dest_dir}")
        
        logging.info(f"Created {directories_created} directories in output structure")
        return directories_created
    
    def find_pdf_files(self, search_criteria: Dict[str, List[str]]) -> Dict[str, List[Path]]:
        """
        Find PDF files categorized by document type based on search criteria.
        
        Args:
            search_criteria: Dictionary mapping document types to search terms
            
        Returns:
            Dictionary mapping document types to lists of PDF file paths
        """
        logging.info("Scanning for PDF files...")
        
        pdfs_by_type = {doc_type: [] for doc_type in search_criteria.keys()}
        ignored_count = 0
        total_pdfs = 0
        
        # Walk through source directory
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    total_pdfs += 1
                    file_path = Path(root) / file
                    
                    # Check if we should ignore paths with directories starting with '#'
                    if self.ignore_hash_paths and self._has_hash_directory(file_path):
                        ignored_count += 1
                        logging.debug(f"Ignored PDF (hash directory): {file_path}")
                        continue
                    
                    # Classify document type based on filename
                    doc_type = self._classify_document(file, search_criteria)
                    # All PDFs will now be classified (either specific type or general_medical_document)
                    pdfs_by_type[doc_type].append(file_path)
                    logging.debug(f"Found {doc_type} PDF: {file_path}")
        
        # Log summary
        logging.info(f"Found {total_pdfs} total PDF files")
        if self.ignore_hash_paths:
            logging.info(f"Ignored {ignored_count} files in directories starting with '#'")
        
        for doc_type, pdfs in pdfs_by_type.items():
            logging.info(f"{doc_type}: {len(pdfs)} files")
        
        return pdfs_by_type
    
    def _has_hash_directory(self, file_path: Path) -> bool:
        """Check if file path contains any directory starting with '#'."""
        # Get all path components except the filename
        path_parts = file_path.relative_to(self.source_dir).parts[:-1]
        return any(part.startswith('#') for part in path_parts)
    
    def _classify_document(self, filename: str, search_criteria: Dict[str, List[str]]) -> str:
        """
        Classify document type based on filename using search criteria.
        
        Args:
            filename: Name of the file to classify
            search_criteria: Dictionary mapping document types to search terms
            
        Returns:
            Document type string, defaults to 'general_medical_document' if no specific match
        """
        filename_lower = filename.lower()
        
        # Check specific document types first (excluding general_medical_document)
        for doc_type, search_terms in search_criteria.items():
            if doc_type == 'general_medical_document':
                continue  # Skip general category for now
                
            if any(term.lower() in filename_lower for term in search_terms):
                return doc_type
        
        # If no specific type matches, classify as general medical document
        # This ensures ALL PDFs are processed
        return 'general_medical_document'
    
    def get_patient_folders(self) -> Dict[str, Set[str]]:
        """
        Get mapping of medical practice folders to patient names.
        
        Returns:
            Dictionary mapping practice names to sets of patient names
        """
        patient_mapping = {}
        
        # Look for Portfolio directories under practice folders
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)
            
            # Check if this is a portfolio directory
            if 'Portfolio #' in root_path.name:
                # Get the practice name (parent directory)
                practice_name = root_path.parent.name
                
                # Initialize practice if not seen before
                if practice_name not in patient_mapping:
                    patient_mapping[practice_name] = set()
                
                # Add patient folders in this portfolio
                for patient_dir in dirs:
                    # Skip directories starting with '#'
                    if not patient_dir.startswith('#'):
                        patient_mapping[practice_name].add(patient_dir)
        
        return patient_mapping