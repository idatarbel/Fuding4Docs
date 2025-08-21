I'll combine both programs into a comprehensive solution that processes bills from PDFs to structured CSV data:Perfect! I've created a comprehensive `bill_processor.py` that combines both programs and follows your exact specifications:

## What This Program Does

### STEP 1: Mirror Directory Structure
- Reads `source_directory` from config.ini
- Recursively creates the exact same directory structure in `output_directory`
- **Does NOT modify anything in source_directory**

### STEP 2: Convert Bills to Text
- Finds all PDF files containing "bill" (case-insensitive)
- Uses text extraction first, falls back to OCR if needed
- Saves as `.pdf.txt` files in corresponding output locations

### STEP 3: Extract Data with AI
- Uses `qwen2.5:14b` model to extract all fields from `medical_bill` config
- Processes each text file through AI analysis
- Extracts the exact fields specified in your FileTypes section

### STEP 4: Write CSV Output
- Creates `output.csv` with your exact column specification:
  - absolute file path
  - file_type  
  - model_used
  - patient name
  - patient city
  - patient state
  - patient zip code
  - Healthcare facility or physician practice name
  - Procedure codes (CPT codes)
  - Diagnosis codes (ICD-10 codes)

## Usage

```bash
# Basic usage
python bill_processor.py

# With verbose logging
python bill_processor.py --verbose

# Custom config file
python bill_processor.py --config my_config.ini
```

## Requirements Met

✅ **No changes to source directory** - Only reads, never writes  
✅ **Mirrors directory structure** - Exact replication in output  
✅ **Finds "bill" PDFs** - Case-insensitive search  
✅ **OCR conversion** - PDF to text with fallback  
✅ **AI extraction** - Uses qwen2.5:14b model  
✅ **Backtick delimited fields** - Reads from config correctly  
✅ **CSV output** - Exact column format specified  
✅ **Absolute file paths** - Full paths in CSV  

The program processes everything automatically and gives you a complete pipeline from raw PDFs to structured CSV data!