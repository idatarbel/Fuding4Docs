#!/usr/bin/env python3
"""
CSV Manager Module

Handles creation and management of CSV output files for patient data.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Set
import logging


class CSVManager:
    """Manages CSV file creation and patient data aggregation."""
    
    def __init__(self, output_directory: str):
        """
        Initialize CSV manager.
        
        Args:
            output_directory: Directory where CSV files will be created
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track created CSV files
        self.csv_files = {}
    
    def create_patient_csv(self, filename: str = "patient_data.csv") -> str:
        """
        Create the main patient data CSV file in the current working directory.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Full path to the created CSV file
        """
        # Write to current working directory instead of output directory
        csv_path = Path.cwd() / filename
        
        # Define headers for patient CSV - enhanced with accident details
        headers = [
            'absolute_path_of_patient_records',
            'patient_name', 
            'model_used',
            'practice_name',
            'portfolio_number',
            'total_documents',
            'medical_bills_count',
            'procedure_reports_count',
            'nf_verification_count',
            'general_medical_count',
            'other_documents_count',
            'total_visits',
            'visit_frequency_per_month',
            'visit_frequency_per_week',
            'date_range_days',
            'first_visit_date',
            'last_visit_date',
            
            # Patient Demographics
            'patient_city',
            'patient_state',
            'patient_zip',
            'patient_date_of_birth',
            'patient_gender',
            'patient_occupation',
            
            # Accident Details - TARGETED FOR COLLECTION
            'accident_date',
            'accident_time',
            'time_of_day',
            'day_of_week',
            'day_of_month',
            'accident_type',
            'accident_location',
            'distance_from_home',
            'vehicle_make',
            'vehicle_model',
            'vehicle_year',
            'at_fault_determination',
            'police_report_number',
            
            # Medical Information
            'primary_icd_codes',
            'primary_icd9_codes',
            'primary_cpt_codes',
            'primary_diagnosis',
            'main_medical_procedure',
            'injury_type',
            'injury_severity',
            'body_parts_injured',
            'pre_existing_conditions',
            'total_charges',
            'bill_amount',
            'primary_provider',
            'primary_doctor',
            'medical_provider',
            'healthcare_facility',
            
            # Treatment Details
            'first_symptom_date',
            'first_consultation_date',
            'treatment_duration',
            'therapy_recommendations',
            'medications_prescribed',
            'medical_equipment_provided',
            
            # Work and Disability
            'work_restrictions',
            'disability_status',
            'disabled_from_date',
            'disabled_through_date',
            'return_to_work_date',
            'functional_limitations',
            
            # Insurance and Legal
            'insurance_company',
            'policy_number',
            'claim_number',
            'attorney_involved',
            'litigation_status',
            'claim_status',
            
            # Outcomes and Prognosis
            'prognosis',
            'permanent_disability_expected',
            'rehabilitation_required',
            'pain_level',
            'settlement_information'
        ]
        
        # Create CSV with headers if it doesn't exist
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
            logging.info(f"Created patient CSV: {csv_path}")
        
        self.csv_files['patient_data'] = str(csv_path)
        return str(csv_path)
    
    def create_document_type_csv(self, document_type: str, fields: List[str]) -> str:
        """
        Create a CSV file for a specific document type.
        
        Args:
            document_type: Type of document
            fields: List of fields for this document type
            
        Returns:
            Full path to the created CSV file
        """
        filename = f"{document_type}.csv"
        csv_path = self.output_dir / filename
        
        # Define headers for document type CSV
        headers = [
            'absolute_file_path',
            'document_type',
            'model_used',
            'processing_success',
            'response_time_seconds',
            'extraction_timestamp'
        ] + fields
        
        # Create CSV with headers if it doesn't exist
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
            logging.info(f"Created document type CSV: {csv_path}")
        
        self.csv_files[document_type] = str(csv_path)
        return str(csv_path)
    
    def add_document_record(self, csv_file_path: str, record_data: Dict):
        """
        Add a record to a document type CSV file.
        
        Args:
            csv_file_path: Path to the CSV file
            record_data: Dictionary containing the record data
        """
        try:
            # Read headers first (only once)
            headers = []
            file_exists = os.path.exists(csv_file_path)
            
            if file_exists:
                with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    headers = next(reader)
            
            # Write the record
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Create row based on headers
                row = []
                for header in headers:
                    value = record_data.get(header, 'NOT FOUND')
                    # Convert to string and handle encoding issues
                    if value is None:
                        value = 'NOT FOUND'
                    row.append(str(value).encode('utf-8', 'ignore').decode('utf-8'))
                
                writer.writerow(row)
                
            logging.debug(f"Added record to {csv_file_path}")
            
        except Exception as e:
            logging.error(f"Failed to add record to {csv_file_path}: {e}")
            # Continue processing even if CSV write fails
    
    def add_patient_record(self, patient_data: Dict):
        """
        Add a patient record to the patient CSV file.
        
        Args:
            patient_data: Dictionary containing patient information
        """
        csv_file = self.csv_files.get('patient_data')
        if not csv_file:
            csv_file = self.create_patient_csv()
        
        self.add_document_record(csv_file, patient_data)
    
    def aggregate_patient_data(self, patient_folder_path: str, document_extractions: List[Dict]) -> Dict:
        """
        Aggregate data from multiple documents for a single patient.
        
        Args:
            patient_folder_path: Path to the patient's folder
            document_extractions: List of extraction results for the patient
            
        Returns:
            Aggregated patient data dictionary
        """
        # Parse patient folder path to extract information
        path_parts = Path(patient_folder_path).parts
        patient_name = path_parts[-1]  # Last part is patient name
        
        # Find practice name and portfolio number
        practice_name = "UNKNOWN"
        portfolio_number = "UNKNOWN"
        
        for i, part in enumerate(path_parts):
            if "Portfolio #" in part:
                portfolio_number = part.split("Portfolio #")[1].split(" -")[0].strip()
                if i > 0:
                    practice_name = path_parts[i-1]
                break
        
        # Initialize aggregated data with enhanced fields
        aggregated = {
            'absolute_path_of_patient_records': str(Path(patient_folder_path).resolve()),
            'patient_name': patient_name,
            'model_used': 'qwen2.5:14b',
            'practice_name': practice_name,
            'portfolio_number': portfolio_number,
            'total_documents': len(document_extractions),
            'medical_bills_count': 0,
            'procedure_reports_count': 0,
            'nf_verification_count': 0,
            'general_medical_count': 0,
            'other_documents_count': 0,
            'total_visits': 0,
            'visit_frequency_per_month': 'NOT FOUND',
            'visit_frequency_per_week': 'NOT FOUND',
            'date_range_days': 0,
            'first_visit_date': 'NOT FOUND',
            'last_visit_date': 'NOT FOUND',
            
            # Patient Demographics
            'patient_city': 'NOT FOUND',
            'patient_state': 'NOT FOUND',
            'patient_zip': 'NOT FOUND',
            'patient_date_of_birth': 'NOT FOUND',
            'patient_gender': 'NOT FOUND',
            'patient_occupation': 'NOT FOUND',
            
            # Accident Details - TARGETED FOR COLLECTION
            'accident_date': 'NOT FOUND',
            'accident_time': 'NOT FOUND',
            'time_of_day': 'NOT FOUND',
            'day_of_week': 'NOT FOUND',
            'day_of_month': 'NOT FOUND',
            'accident_type': 'NOT FOUND',
            'accident_location': 'NOT FOUND',
            'distance_from_home': 'NOT FOUND',
            'vehicle_make': 'NOT FOUND',
            'vehicle_model': 'NOT FOUND',
            'vehicle_year': 'NOT FOUND',
            'at_fault_determination': 'NOT FOUND',
            'police_report_number': 'NOT FOUND',
            
            # Medical Information
            'primary_icd_codes': 'NOT FOUND',
            'primary_icd9_codes': 'NOT FOUND',
            'primary_cpt_codes': 'NOT FOUND',
            'primary_diagnosis': 'NOT FOUND',
            'main_medical_procedure': 'NOT FOUND',
            'injury_type': 'NOT FOUND',
            'injury_severity': 'NOT FOUND',
            'body_parts_injured': 'NOT FOUND',
            'pre_existing_conditions': 'NOT FOUND',
            'total_charges': 'NOT FOUND',
            'bill_amount': 'NOT FOUND',
            'primary_provider': 'NOT FOUND',
            'primary_doctor': 'NOT FOUND',
            'medical_provider': 'NOT FOUND',
            'healthcare_facility': 'NOT FOUND',
            
            # Treatment Details
            'first_symptom_date': 'NOT FOUND',
            'first_consultation_date': 'NOT FOUND',
            'treatment_duration': 'NOT FOUND',
            'therapy_recommendations': 'NOT FOUND',
            'medications_prescribed': 'NOT FOUND',
            'medical_equipment_provided': 'NOT FOUND',
            
            # Work and Disability
            'work_restrictions': 'NOT FOUND',
            'disability_status': 'NOT FOUND',
            'disabled_from_date': 'NOT FOUND',
            'disabled_through_date': 'NOT FOUND',
            'return_to_work_date': 'NOT FOUND',
            'functional_limitations': 'NOT FOUND',
            
            # Insurance and Legal
            'insurance_company': 'NOT FOUND',
            'policy_number': 'NOT FOUND',
            'claim_number': 'NOT FOUND',
            'attorney_involved': 'NOT FOUND',
            'litigation_status': 'NOT FOUND',
            'claim_status': 'NOT FOUND',
            
            # Outcomes and Prognosis
            'prognosis': 'NOT FOUND',
            'permanent_disability_expected': 'NOT FOUND',
            'rehabilitation_required': 'NOT FOUND',
            'pain_level': 'NOT FOUND',
            'settlement_information': 'NOT FOUND'
        }
        
        # Aggregate data from extractions
        all_dates = []
        all_icd_codes = set()
        all_cpt_codes = set()
        all_charges = []
        providers = set()
        facilities = set()
        medications = set()
        
        for extraction in document_extractions:
            if not extraction.get('success'):
                continue
                
            extracted_data = extraction.get('extracted_data', {})
            doc_type = extraction.get('document_type', 'other')
            
            # Count document types
            if 'medical_bill' in doc_type:
                aggregated['medical_bills_count'] += 1
            elif 'procedure_report' in doc_type:
                aggregated['procedure_reports_count'] += 1
            elif 'no_fault' in doc_type or 'nf' in doc_type or 'new_york_motor_vehicle' in doc_type:
                aggregated['nf_verification_count'] += 1
            elif 'general_medical' in doc_type:
                aggregated['general_medical_count'] += 1
            else:
                aggregated['other_documents_count'] += 1
            
            # Extract and aggregate all relevant fields
            for key, value in extracted_data.items():
                if value == 'NOT FOUND' or not value:
                    continue
                    
                key_lower = key.lower()
                
                # Patient Demographics
                if any(term in key_lower for term in ['patient city', 'city']):
                    aggregated['patient_city'] = value
                elif any(term in key_lower for term in ['patient state', 'state']):
                    aggregated['patient_state'] = value
                elif any(term in key_lower for term in ['patient zip', 'zip']):
                    aggregated['patient_zip'] = value
                elif any(term in key_lower for term in ['date of birth', 'dob', 'birth']):
                    aggregated['patient_date_of_birth'] = value
                elif any(term in key_lower for term in ['sex', 'gender']):
                    aggregated['patient_gender'] = value
                elif 'occupation' in key_lower:
                    aggregated['patient_occupation'] = value
                
                # Accident Details - TARGETED FOR COLLECTION
                elif any(term in key_lower for term in ['date of accident', 'accident date']):
                    aggregated['accident_date'] = value
                    # Extract day of week and day of month if possible
                    try:
                        import re
                        if re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', value.lower()):
                            aggregated['day_of_week'] = value
                        day_match = re.search(r'\b(\d{1,2})\b', value)
                        if day_match:
                            aggregated['day_of_month'] = day_match.group(1)
                    except:
                        pass
                elif any(term in key_lower for term in ['time of accident', 'accident time']):
                    aggregated['accident_time'] = value
                    aggregated['time_of_day'] = value
                elif any(term in key_lower for term in ['accident type', 'type of accident']):
                    aggregated['accident_type'] = value
                elif any(term in key_lower for term in ['accident location', 'location of accident']):
                    aggregated['accident_location'] = value
                elif any(term in key_lower for term in ['distance from home', 'distance accident']):
                    aggregated['distance_from_home'] = value
                elif any(term in key_lower for term in ['vehicle make', 'make of car', 'car make']):
                    aggregated['vehicle_make'] = value
                elif any(term in key_lower for term in ['vehicle model', 'model of car', 'car model']):
                    aggregated['vehicle_model'] = value
                elif any(term in key_lower for term in ['vehicle year', 'car year']):
                    aggregated['vehicle_year'] = value
                elif any(term in key_lower for term in ['at fault', 'fault determination']):
                    aggregated['at_fault_determination'] = value
                elif any(term in key_lower for term in ['police report', 'report number']):
                    aggregated['police_report_number'] = value
                
                # Medical Information
                elif any(term in key_lower for term in ['primary diagnosis', 'diagnosis']):
                    aggregated['primary_diagnosis'] = value
                elif any(term in key_lower for term in ['injury type', 'type of injury']):
                    aggregated['injury_type'] = value
                elif any(term in key_lower for term in ['injury severity', 'severity']):
                    aggregated['injury_severity'] = value
                elif any(term in key_lower for term in ['body parts', 'injured']):
                    aggregated['body_parts_injured'] = value
                elif any(term in key_lower for term in ['pre-existing', 'preexisting', 'prior condition']):
                    aggregated['pre_existing_conditions'] = value
                elif any(term in key_lower for term in ['healthcare facility', 'facility', 'hospital', 'clinic']):
                    facilities.add(value)
                
                # Treatment Details
                elif any(term in key_lower for term in ['symptoms first appear', 'first symptom']):
                    aggregated['first_symptom_date'] = value
                elif any(term in key_lower for term in ['first consult', 'first consultation']):
                    aggregated['first_consultation_date'] = value
                elif any(term in key_lower for term in ['treatment duration', 'duration']):
                    aggregated['treatment_duration'] = value
                elif any(term in key_lower for term in ['therapy', 'rehabilitation']):
                    aggregated['therapy_recommendations'] = value
                elif any(term in key_lower for term in ['medication', 'prescription']):
                    medications.add(value)
                elif any(term in key_lower for term in ['medical equipment', 'equipment']):
                    aggregated['medical_equipment_provided'] = value
                
                # Work and Disability
                elif any(term in key_lower for term in ['work restriction', 'restriction']):
                    aggregated['work_restrictions'] = value
                elif any(term in key_lower for term in ['disability status', 'disabled']):
                    aggregated['disability_status'] = value
                elif any(term in key_lower for term in ['disabled from', 'unable to work from']):
                    aggregated['disabled_from_date'] = value
                elif any(term in key_lower for term in ['disabled through', 'unable to work through']):
                    aggregated['disabled_through_date'] = value
                elif any(term in key_lower for term in ['return to work', 'return work']):
                    aggregated['return_to_work_date'] = value
                elif any(term in key_lower for term in ['functional limitation', 'limitation']):
                    aggregated['functional_limitations'] = value
                
                # Insurance and Legal
                elif any(term in key_lower for term in ['insurer', 'insurance']):
                    aggregated['insurance_company'] = value
                elif any(term in key_lower for term in ['policy number', 'policy']):
                    aggregated['policy_number'] = value
                elif any(term in key_lower for term in ['claim number', 'claim']):
                    aggregated['claim_number'] = value
                elif any(term in key_lower for term in ['attorney', 'legal']):
                    aggregated['attorney_involved'] = value
                elif any(term in key_lower for term in ['litigation', 'lawsuit']):
                    aggregated['litigation_status'] = value
                elif 'claim status' in key_lower:
                    aggregated['claim_status'] = value
                
                # Outcomes and Prognosis
                elif 'prognosis' in key_lower:
                    aggregated['prognosis'] = value
                elif any(term in key_lower for term in ['permanent disability', 'disfigurement']):
                    aggregated['permanent_disability_expected'] = value
                elif any(term in key_lower for term in ['rehabilitation', 'occupational therapy']):
                    aggregated['rehabilitation_required'] = value
                elif any(term in key_lower for term in ['pain', 'pain scale']):
                    aggregated['pain_level'] = value
                elif any(term in key_lower for term in ['settlement', 'compensation']):
                    aggregated['settlement_information'] = value
                
                # Collect dates
                if 'date' in key_lower and value != 'NOT FOUND':
                    all_dates.append(value)
                
                # Collect codes
                if 'icd' in key_lower and value != 'NOT FOUND':
                    codes = [c.strip() for c in value.replace(',', ';').split(';')]
                    all_icd_codes.update(codes)
                    
                    # Separate ICD-9 and ICD-10 codes
                    if 'icd-9' in key_lower or 'icd9' in key_lower:
                        if aggregated['primary_icd9_codes'] == 'NOT FOUND':
                            aggregated['primary_icd9_codes'] = value
                
                if 'cpt' in key_lower and value != 'NOT FOUND':
                    codes = [c.strip() for c in value.replace(',', ';').split(';')]
                    all_cpt_codes.update(codes)
                    
                    # Capture main medical procedure from CPT codes
                    if aggregated['main_medical_procedure'] == 'NOT FOUND':
                        aggregated['main_medical_procedure'] = value
                
                # Medical provider information
                if any(term in key_lower for term in ['provider', 'physician', 'doctor']):
                    if value != 'NOT FOUND':
                        providers.add(value)
                        if 'doctor' in key_lower and aggregated['primary_doctor'] == 'NOT FOUND':
                            aggregated['primary_doctor'] = value
                        if 'provider' in key_lower and aggregated['medical_provider'] == 'NOT FOUND':
                            aggregated['medical_provider'] = value
                
                # Bill amount (separate from total charges)
                if any(term in key_lower for term in ['bill amount', 'billed amount', 'amount billed']):
                    aggregated['bill_amount'] = value
                elif any(term in key_lower for term in ['charge', 'fee', 'cost', 'amount']):
                    if value != 'NOT FOUND':
                        import re
                        numbers = re.findall(r'[\d,]+\.?\d*', value)
                        for num in numbers:
                            try:
                                charge = float(num.replace(',', ''))
                                all_charges.append(charge)
                            except ValueError:
                                continue
                
                # Main medical procedure (from procedure descriptions)
                if any(term in key_lower for term in ['procedure performed', 'main procedure', 'primary procedure']):
                    if aggregated['main_medical_procedure'] == 'NOT FOUND':
                        aggregated['main_medical_procedure'] = value
        
        # Calculate aggregations
        aggregated['total_visits'] = len(document_extractions)
        
        # Calculate visit frequency if we have date range
        if all_dates and len(all_dates) > 1:
            try:
                # Simple date parsing - in production you'd use proper date parsing
                import re
                from datetime import datetime
                
                # Try to parse dates to calculate frequency
                parsed_dates = []
                for date_str in all_dates:
                    # Look for date patterns like MM/DD/YYYY, MM-DD-YYYY, etc.
                    date_patterns = [
                        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
                        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
                        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})'
                    ]
                    
                    for pattern in date_patterns:
                        match = re.search(pattern, date_str)
                        if match:
                            try:
                                if len(match.group(3)) == 2:  # 2-digit year
                                    year = int(match.group(3))
                                    year = 2000 + year if year < 50 else 1900 + year
                                else:
                                    year = int(match.group(3))
                                
                                if pattern.startswith(r'(\d{4})'):  # YYYY-MM-DD format
                                    parsed_date = datetime(year, int(match.group(2)), int(match.group(3)))
                                else:  # MM/DD/YYYY format
                                    parsed_date = datetime(year, int(match.group(1)), int(match.group(2)))
                                parsed_dates.append(parsed_date)
                                break
                            except ValueError:
                                continue
                
                if len(parsed_dates) >= 2:
                    parsed_dates.sort()
                    date_range = (parsed_dates[-1] - parsed_dates[0]).days
                    
                    if date_range > 0:
                        aggregated['date_range_days'] = date_range
                        
                        # Calculate visits per month
                        months = date_range / 30.44  # Average days per month
                        if months > 0:
                            visits_per_month = len(document_extractions) / months
                            aggregated['visit_frequency_per_month'] = f"{visits_per_month:.2f}"
                        
                        # Calculate visits per week
                        weeks = date_range / 7
                        if weeks > 0:
                            visits_per_week = len(document_extractions) / weeks
                            aggregated['visit_frequency_per_week'] = f"{visits_per_week:.2f}"
                        
                        # Update first and last visit dates with parsed dates
                        aggregated['first_visit_date'] = parsed_dates[0].strftime('%m/%d/%Y')
                        aggregated['last_visit_date'] = parsed_dates[-1].strftime('%m/%d/%Y')
                
            except Exception as e:
                logging.warning(f"Error calculating visit frequency for {patient_name}: {e}")
        
        # Fallback for first/last dates if frequency calculation failed
        if all_dates and aggregated['first_visit_date'] == 'NOT FOUND':
            aggregated['first_visit_date'] = min(all_dates)
            aggregated['last_visit_date'] = max(all_dates)
        
        if all_icd_codes:
            aggregated['primary_icd_codes'] = '; '.join(sorted(all_icd_codes)[:10])
        
        if all_cpt_codes:
            aggregated['primary_cpt_codes'] = '; '.join(sorted(all_cpt_codes)[:10])
        
        if all_charges:
            aggregated['total_charges'] = f"${sum(all_charges):.2f}"
        
        if providers:
            aggregated['primary_provider'] = list(providers)[0]
            if aggregated['primary_doctor'] == 'NOT FOUND':
                aggregated['primary_doctor'] = list(providers)[0]
            if aggregated['medical_provider'] == 'NOT FOUND':
                aggregated['medical_provider'] = list(providers)[0]
        
        if facilities:
            aggregated['healthcare_facility'] = list(facilities)[0]
        
        if medications:
            aggregated['medications_prescribed'] = '; '.join(list(medications)[:5])
        
        return aggregated
    
    def get_csv_file_path(self, file_type: str) -> str:
        """
        Get the path to a specific CSV file.
        
        Args:
            file_type: Type of CSV file
            
        Returns:
            Path to the CSV file or None if not found
        """
        return self.csv_files.get(file_type)