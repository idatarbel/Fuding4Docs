#!/usr/bin/env python3
"""
AI Data Extractor Module

Handles AI-powered data extraction from text documents using Ollama.
"""

import json
import time
from typing import Dict, List
import logging
import requests


class AIExtractor:
    """Extracts structured data from text documents using AI models."""
    
    def __init__(self, model_name: str = "qwen2.5:14b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize AI extractor.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama service
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._verify_model_available()
    
    def _verify_model_available(self):
        """Verify that the required model is available in Ollama."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            available_models = [m['name'] for m in models]
            
            # Check if our model is available
            model_found = any(self.model_name in model for model in available_models)
            
            if not model_found:
                raise RuntimeError(f"Model {self.model_name} not found. Available models: {available_models}")
            
            logging.info(f"AI model {self.model_name} is available")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}: {e}")
    
    def extract_data(self, text_content: str, document_type: str, expected_fields: List[str]) -> Dict:
        """
        Extract structured data from text content.
        
        Args:
            text_content: Text content to extract data from
            document_type: Type of document being processed
            expected_fields: List of fields to extract
            
        Returns:
            Dictionary with extraction results
        """
        # Keep larger content size for comprehensive extraction
        if len(text_content) > 12000:  # Increased for more thorough analysis
            text_content = text_content[:12000] + "\n[Content truncated for processing...]"
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(text_content, document_type, expected_fields)
        
        # Query the model
        start_time = time.time()
        try:
            ai_response = self._query_model(prompt)
            response_time = time.time() - start_time
            
            # Parse the response
            extracted_data = self._parse_extraction_response(ai_response, expected_fields)
            
            return {
                'success': True,
                'extracted_data': extracted_data,
                'raw_response': ai_response,
                'response_time': response_time,
                'model_used': self.model_name
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            logging.error(f"AI extraction failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time,
                'model_used': self.model_name
            }
    
    def _create_extraction_prompt(self, content: str, document_type: str, fields: List[str]) -> str:
        """
        Create a prompt for extracting document information.
        
        Args:
            content: Document content
            document_type: Type of document
            fields: List of fields to extract
            
        Returns:
            Formatted prompt string
        """
        fields_list = '\n'.join([f"- {field}" for field in fields])
        
        # Special handling for general medical documents
        if document_type == 'general_medical_document':
            prompt = f"""TASK: You are analyzing a medical document to extract any relevant information that could be useful for insurance lawsuit outcome prediction. This document may be any type of medical record, form, note, report, or correspondence.

DOCUMENT CONTENT:
{content}

EXTRACT ANY AVAILABLE INFORMATION FOR THESE FIELDS:
{fields_list}

INSTRUCTIONS:
1. Carefully read the entire document to understand what type of medical document this is
2. Extract any information that matches or relates to the fields listed above
3. If a field cannot be found or determined, write "NOT FOUND"
4. Extract only factual information directly stated in the document
5. Do not infer or guess values - only extract what is explicitly written
6. For dates, use the format found in the document
7. For codes (ICD, CPT), include all relevant codes found
8. Pay special attention to information related to:
   - Patient demographics and contact information
   - Accident or injury details
   - Medical treatments and procedures
   - Billing and insurance information
   - Provider information
   - Dates of service
   - Diagnoses and procedures codes
9. Return only the field name and value, no explanations

FORMAT YOUR RESPONSE AS:
Field Name: Extracted Value
(One field per line)

BEGIN EXTRACTION:"""
        else:
            # Use original prompt for specific document types
            prompt = f"""TASK: Extract specific information from this {document_type.replace('_', ' ')} document.

DOCUMENT CONTENT:
{content}

EXTRACT THESE EXACT FIELDS:
{fields_list}

INSTRUCTIONS:
1. For each field listed above, find the exact value from the document
2. If a field cannot be found or determined, write "NOT FOUND"
3. Extract only factual information directly stated in the document
4. Do not infer or guess values
5. For dates, use the format found in the document
6. For codes (ICD, CPT), include all relevant codes found
7. Return only the field name and value, no explanations

FORMAT YOUR RESPONSE AS:
Field Name: Extracted Value
(One field per line)

BEGIN EXTRACTION:"""
        
        return prompt
    
    def _query_model(self, prompt: str) -> str:
        """
        Query the Ollama model with the given prompt.
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            Model response text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 400,     # Balanced response length
                "num_ctx": 3072,        # Reduced context for faster processing on GPU
                "top_k": 8,
                "top_p": 0.85,
                "num_batch": 8,         # Batch processing for GPU efficiency
                "num_gpu": 1,           # Use 1 GPU
                "stop": ["Human:", "Assistant:", "TASK:"]
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Reasonable timeout for GPU processing
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.Timeout:
            raise Exception("Model query timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error querying model: {e}")
    
    def _parse_extraction_response(self, response: str, expected_fields: List[str]) -> Dict[str, str]:
        """
        Parse the model's extraction response into structured data.
        
        Args:
            response: Raw response from the model
            expected_fields: List of expected field names
            
        Returns:
            Dictionary mapping field names to extracted values
        """
        extracted = {}
        
        # Initialize all fields as "NOT FOUND"
        for field in expected_fields:
            extracted[field] = "NOT FOUND"
        
        # Parse response line by line
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines or lines without colons
            if not line or ':' not in line:
                continue
            
            # Split on first colon
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            field_value = parts[1].strip()
            
            # Find matching expected field (case insensitive, flexible matching)
            for expected_field in expected_fields:
                if self._fields_match(field_name, expected_field):
                    # Only update if we found a meaningful value
                    if field_value and field_value.upper() not in ['NOT FOUND', 'N/A', 'NULL', 'NONE', 'UNKNOWN']:
                        extracted[expected_field] = field_value
                    break
        
        return extracted
    
    def _fields_match(self, response_field: str, expected_field: str) -> bool:
        """
        Check if a response field matches an expected field.
        
        Args:
            response_field: Field name from model response
            expected_field: Expected field name
            
        Returns:
            True if fields match, False otherwise
        """
        response_lower = response_field.lower().strip()
        expected_lower = expected_field.lower().strip()
        
        # Exact match
        if response_lower == expected_lower:
            return True
        
        # Check if response field is contained in expected field
        if response_lower in expected_lower:
            return True
        
        # Check if expected field is contained in response field
        if expected_lower in response_lower:
            return True
        
        # Check for key word matches
        response_words = set(response_lower.split())
        expected_words = set(expected_lower.split())
        
        # If more than half the words match, consider it a match
        if len(response_words & expected_words) >= min(len(response_words), len(expected_words)) * 0.5:
            return True
        
        return False
    
    def save_extraction_data(self, extraction_result: Dict, output_file_path: str):
        """
        Save extraction results to a data file.
        
        Args:
            extraction_result: Result from extract_data method
            output_file_path: Path to save the extraction data
        """
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write("EXTRACTION RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Model Used: {extraction_result.get('model_used', 'Unknown')}\n")
                f.write(f"Response Time: {extraction_result.get('response_time', 0):.2f} seconds\n")
                f.write(f"Success: {extraction_result.get('success', False)}\n\n")
                
                if extraction_result.get('success'):
                    f.write("EXTRACTED DATA:\n")
                    f.write("-" * 30 + "\n")
                    
                    extracted_data = extraction_result.get('extracted_data', {})
                    for field, value in extracted_data.items():
                        f.write(f"{field}: {value}\n")
                    
                    f.write("\nRAW AI RESPONSE:\n")
                    f.write("-" * 30 + "\n")
                    f.write(extraction_result.get('raw_response', 'No response'))
                else:
                    f.write(f"ERROR: {extraction_result.get('error', 'Unknown error')}\n")
                
            logging.info(f"Saved extraction data to: {output_file_path}")
            
        except Exception as e:
            logging.error(f"Failed to save extraction data to {output_file_path}: {e}")