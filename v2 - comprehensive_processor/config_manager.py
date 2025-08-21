#!/usr/bin/env python3
"""
Configuration Management Module

Handles loading and validation of configuration settings from config.ini file.
"""

import os
import configparser
import sys
from typing import Dict, List
import logging


class ConfigManager:
    """Manages application configuration from config.ini file."""
    
    def __init__(self, config_file: str = "config.ini"):
        """Initialize configuration manager with config file path."""
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> configparser.ConfigParser:
        """Load and validate configuration from config.ini file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file '{self.config_file}' not found.")
        
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        # Validate required sections
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: configparser.ConfigParser):
        """Validate configuration has required sections and values."""
        # Check for required sections
        required_sections = ['General']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Config file must contain a [{section}] section")
        
        # Validate General section
        general = config['General']
        source_dir = general.get('source_directory', '').strip()
        output_dir = general.get('output_directory', '').strip()
        
        if not source_dir or not output_dir:
            raise ValueError("source_directory and output_directory must be specified in config.ini")
        
        # Check if source directory exists
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    
    def get_general_config(self) -> Dict:
        """Get general configuration settings."""
        general = self.config['General']
        return {
            'source_directory': general.get('source_directory', '').strip(),
            'output_directory': general.get('output_directory', '').strip(),
            'log_directory': general.get('log_directory', './logs').strip(),
            'model_name': general.get('model_name', 'qwen2.5:14b').strip(),
            'ollama_url': general.get('ollama_url', 'http://localhost:11434').strip(),
            'verbose': general.getboolean('verbose', fallback=False),
            'ignore_hash_paths': general.getboolean('ignore_hash_paths', fallback=True)
        }
    
    def get_document_field_definitions(self) -> Dict[str, List[str]]:
        """Get field definitions for each document type."""
        field_definitions = {}
        
        # Medical bill fields - from your specification
        medical_bill_fields = [
            'patient name',
            'patient city',
            'patient state',
            'patient zip code',
            'Healthcare facility or physician practice name',
            'Procedure codes (CPT codes)',
            'Diagnosis codes (ICD-10 codes)'
        ]
        field_definitions['medical_bill'] = medical_bill_fields
        
        # NY Motor Vehicle No Fault Treatment Verification fields - from your specification
        nf_verification_fields = [
            "Provider's name and address",
            "Name and address of insurer or self-insurer",
            "Name, address, and phone number of insurer's claims representative",
            "Policyholder",
            "Policy number",
            "Date of accident",
            "Claim number",
            "Patient's name and address",
            "Date of birth",
            "Sex",
            "Occupation (if known)",
            "Diagnosis and concurrent conditions",
            "When did symptoms first appear? (Date)",
            "When did patient first consult you for this condition? (Date)",
            "Has patient ever had same or similar condition? (Yes/No)",
            "If yes, state when and describe",
            "Is condition solely a result of this automobile accident? (Yes/No)",
            "If no, explain",
            "Is condition due to injury arising out of patient's employment? (Yes/No)",
            "Will injury result in significant disfigurement or permanent disability? (Yes/No/Not determinable at this time)",
            "If yes, describe",
            "Patient was disabled (unable to work) from (Date)",
            "Patient was disabled (unable to work) through (Date)",
            "If still disabled the patient should be able to return to work on (Date)",
            "Will the patient require rehabilitation and/or occupational therapy as a result of the injuries sustained in this accident? (Yes/No)",
            "If yes, describe your recommendation",
            "Date of service",
            "Place of service including zip code",
            "Description of treatment or health service rendered",
            "Fee schedule",
            "Treatment code",
            "Charges",
            "Total charges to date",
            "Treating provider's name",
            "Treating provider's title",
            "License or certification number",
            "Business relationship (Employee/Independent contractor/Other)",
            "If treating provider is different than billing provider complete the following",
            "If the provider of service is a professional service corporation or doing business under an assumed name (DBA), list the owner and professional licensing credentials of all owners",
            "Is patient still under your care for this condition? (Yes/No)",
            "Estimated duration of future treatment",
            "Authorization to pay benefits checkbox",
            "Assignment of no-fault benefits checkbox",
            "Has an original authorization or assignment previously been executed? (Yes/No)",
            "Is the original signature of the parties on file? (Yes/No)",
            "IRS/TIN identification number",
            "WCB rating code",
            "If none, specialty"
        ]
        field_definitions['new_york_motor_vehicle_no_fault_treatment_verification'] = nf_verification_fields
        
        # Procedure report fields - from your specification
        procedure_report_fields = [
            "Patient name, date of birth, and medical record number",
            "Date and time of procedure",
            "Location where procedure was performed",
            "Pre-procedure diagnosis or indication for the procedure",
            "Post-procedure diagnosis or findings",
            "Patient's medical history relevant to the procedure",
            "Current medications and allergies",
            "Name and type of procedure performed",
            "CPT or procedure codes",
            "Detailed description of technique used",
            "Equipment, instruments, or devices utilized",
            "Duration of the procedure",
            "Primary physician/surgeon performing the procedure",
            "Assisting physicians or specialists",
            "Anesthesiologist",
            "Nursing staff and technicians involved",
            "Anatomical findings observed during the procedure",
            "Tissue samples collected for pathology",
            "Measurements, images, or test results obtained",
            "Any complications or unexpected findings",
            "Medications administered during the procedure",
            "Anesthesia type and dosage",
            "Implants, grafts, or prosthetics used",
            "Surgical technique modifications made",
            "Patient's condition immediately following procedure",
            "Post-procedure instructions and care plan",
            "Follow-up appointment recommendations",
            "Restrictions or activity limitations",
            "Prescribed medications or treatments"
        ]
        field_definitions['procedure_report'] = procedure_report_fields
        
        # General medical document fields - STREAMLINED for speed and key lawsuit prediction factors
        general_medical_fields = [
            # Essential Patient Info (10 fields)
            'Patient name',
            'Patient date of birth',
            'Patient city',
            'Patient state', 
            'Patient zip code',
            'Patient phone number',
            'Medical record number',
            'Patient gender/sex',
            'Patient occupation',
            'Insurance policy number',
            
            # Key Accident Details (8 fields) - CRITICAL FOR LAWSUIT PREDICTION
            'Date of accident',
            'Time of accident',
            'Accident type',
            'Motor vehicle accident details',
            'Vehicle make',
            'Vehicle model',
            'Distance from home to accident',
            'At fault determination',
            
            # Critical Medical Information (15 fields)
            'Date of service',
            'Provider name',
            'Healthcare facility name',
            'Primary diagnosis',
            'Diagnosis codes (ICD-9)',
            'Diagnosis codes (ICD-10)',
            'Procedure codes (CPT codes)',
            'Chief complaint',
            'Treatment provided',
            'Medications prescribed',
            'Pain level',
            'Injury severity',
            'Body parts injured',
            'Pre-existing conditions',
            'Work restrictions',
            
            # Financial and Legal (7 fields) - KEY FOR LAWSUIT OUTCOMES
            'Total charges',
            'Amount billed',
            'Insurance company name',
            'Claim number',
            'Attorney involved',
            'Claim status',
            'Settlement information'
        ]
        field_definitions['general_medical_document'] = general_medical_fields
        
        return field_definitions
    
    def get_document_search_criteria(self) -> Dict[str, List[str]]:
        """Get search criteria for identifying document types."""
        search_criteria = {
            'medical_bill': ['bill', 'invoice', 'statement', 'charge'],
            'new_york_motor_vehicle_no_fault_treatment_verification': ['no fault', 'nf', 'motor vehicle', 'treatment verification'],
            'procedure_report': ['operative', 'procedure', 'surgery', 'operation'],
            'general_medical_document': ['medical', 'patient', 'treatment', 'doctor', 'physician', 'clinic', 'hospital', 'therapy', 'diagnosis', 'prescription', 'medication', 'report', 'note', 'form', 'record', 'chart', 'assessment', 'evaluation', 'consultation', 'discharge', 'admission', 'referral', 'lab', 'test', 'x-ray', 'mri', 'ct', 'scan', 'imaging', 'pathology', 'radiology', 'physical', 'occupational', 'chiropractic', 'orthopedic', 'neurological', 'pain', 'injury', 'accident', 'emergency', 'urgent', 'care', 'wellness', 'health', 'rehabilitation', 'recovery', 'follow-up', 'progress']
        }
        return search_criteria