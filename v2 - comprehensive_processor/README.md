# Medical Document Processing System

A modularized Python application that processes medical documents from PDF files to extract structured data for insurance lawsuit outcome prediction. This system is designed for law firms representing insurance companies in overbilling cases following car accidents.

## Overview

The system processes various types of medical documents including medical bills, insurance claim forms, Assignment of Benefits forms, operative reports, New York State insurance no-fault treatment verification forms, and other medical documentation. It extracts key information to help predict lawsuit outcomes using machine learning.

## Features

- **Directory Structure Mirroring**: Safely mirrors source directory structure without modifying original files
- **Intelligent PDF Classification**: Automatically identifies document types based on filename patterns
- **Hybrid Text Extraction**: Uses both direct text extraction and OCR for maximum accuracy
- **AI-Powered Data Extraction**: Leverages qwen2.5:14b model for structured data extraction
- **Patient Data Aggregation**: Combines multiple documents per patient for comprehensive analysis
- **CSV Output**: Generates machine learning-ready datasets
- **Robust Error Handling**: Continues processing even when individual documents fail
- **Comprehensive Logging**: Detailed logs for monitoring and debugging

## System Requirements

### Software Dependencies
- Python 3.8 or higher
- Tesseract OCR engine
- Ollama with qwen2.5:14b model

### Hardware Recommendations
- 8GB+ RAM (for AI model)
- SSD storage (for faster processing)
- Multi-core CPU (for parallel processing)

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to system PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Install and Setup Ollama

1. Install Ollama from: https://ollama.ai/
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the required model:
   ```bash
   ollama pull qwen2.5:14b
   ```

## Configuration

Create a `config.ini` file in the project directory:

```ini
[General]
# Source directory containing the medical practice folders
source_directory = F:\Medical_Documents

# Output directory where processed files will be saved
output_directory = F:\Medical_Documents_Output

# Directory for log files
log_directory = ./logs

# AI model name (must be available in Ollama)
model_name = qwen2.5:14b

# Ollama service URL
ollama_url = http://localhost:11434

# Enable verbose logging (true/false)
verbose = false

# Ignore paths containing directories that start with '#' (true/false)
ignore_hash_paths = true
```

## Usage

### Basic Usage
```bash
python main_processor.py
```

### With Custom Config File
```bash
python main_processor.py --config my_config.ini
```

### With Verbose Logging
```bash
python main_processor.py --verbose
```

## File Structure

The application expects the following source directory structure:

```
F:\Medical_Documents\
├── Practice Name 1\
│   ├── Portfolio #1 - Practice Name 1\
│   │   ├── Patient_LastName, FirstName\
│   │   │   ├── medical_bill_001.pdf
│   │   │   ├── procedure_report_001.pdf
│   │   │   └── nf_verification_001.pdf
│   │   └── Another_Patient, Name\
│   └── Portfolio #2 - Practice Name 1\
└── Practice Name 2\
    └── Portfolio #1 - Practice Name 2\
```

**Note**: Directories starting with '#' are ignored during processing.

## Output Files

### CSV Files Generated

1. **patient_data.csv**: Aggregated patient information
   - Patient demographics
   - Document counts by type
   - Visit frequency and date ranges
   - Primary ICD/CPT codes
   - Total charges
   - Practice and provider information

2. **medical_bill.csv**: Individual medical bill extractions
3. **procedure_report.csv**: Individual procedure report extractions
4. **new_york_motor_vehicle_no_fault_treatment_verification.csv**: No-fault verification form extractions

### Additional Output

- **Text files**: OCR-converted versions of all PDFs (.pdf.txt)
- **Data files**: Detailed extraction results for each document (.txt__data.txt)
- **Log files**: Processing logs in the specified log directory

## Module Architecture

The application is organized into logical modules:

### `config_manager.py`
- Handles configuration file loading and validation
- Manages document type field definitions
- Provides search criteria for document classification

### `file_scanner.py`
- Mirrors directory structure
- Finds and classifies PDF files
- Manages patient folder mapping

### `pdf_converter.py`
- Converts PDFs to text using hybrid approach
- Handles both direct text extraction and OCR
- Optimized for processing speed and accuracy

### `ai_extractor.py`
- Interfaces with Ollama AI service
- Creates extraction prompts
- Parses AI responses into structured data
- Saves detailed extraction results

### `csv_manager.py`
- Creates and manages CSV output files
- Aggregates patient data from multiple documents
- Handles data formatting and validation

### `main_processor.py`
- Orchestrates the complete processing pipeline
- Provides command-line interface
- Manages error handling and progress reporting

## Document Types Processed

### Medical Bills
Extracts: Patient demographics, facility information, procedure codes (CPT), diagnosis codes (ICD-10)

### New York State Motor Vehicle No-Fault Treatment Verification
Extracts: Comprehensive accident and treatment information including provider details, accident date, patient condition, treatment history, and billing information

### Procedure Reports
Extracts: Detailed surgical/procedure information including patient data, procedure details, findings, complications, and post-procedure care

### Other Documents
The system can process additional document types by extracting relevant information related to patient medical status and billing.

## Data Fields for Machine Learning

The system targets extraction of fields correlated with lawsuit outcomes:

**Patient Information:**
- State, City, Zip code
- Injury type and severity

**Accident Details:**
- Date and type of accident
- Distance from home
- Vehicle information
- Time and day details

**Medical Information:**
- Number and frequency of visits
- Primary procedures and diagnoses
- ICD-9/ICD-10 and CPT codes
- Total billing amounts
- Provider information

## Performance Considerations

- **Processing Speed**: Approximately 30-60 seconds per document depending on size and complexity
- **Memory Usage**: 2-4GB during AI processing
- **Storage**: Plan for 2-3x source directory size for output files
- **Batch Processing**: Can handle hundreds of patients with thousands of documents

## Troubleshooting

### Common Issues

1. **"Tesseract not found"**
   - Ensure Tesseract is installed and in system PATH
   - Test with: `tesseract --version`

2. **"Cannot connect to Ollama"**
   - Start Ollama service: `ollama serve`
   - Verify model is available: `ollama list`

3. **"qwen2.5:14b model not found"**
   - Pull the model: `ollama pull qwen2.5:14b`

4. **Memory errors during processing**
   - Reduce max_pages in PDFConverter
   - Process smaller batches
   - Increase system RAM

### Log Analysis
Check log files in the configured log directory for detailed error information and processing statistics.

## Security and Compliance

- **Data Protection**: No source files are modified or deleted
- **HIPAA Considerations**: Ensure output directory has appropriate access controls
- **Audit Trail**: Comprehensive logging for compliance requirements

## Future Enhancements

- Parallel processing for improved performance
- Additional document type support
- Integration with cloud AI services
- Real-time processing capabilities
- Advanced data validation and quality checks

## License

This software is proprietary and intended for use by authorized legal professionals only.

## Support

For technical support or feature requests, please contact the development team with:
- Detailed error descriptions
- Log file excerpts
- Sample document types (with PHI removed)
- System configuration information