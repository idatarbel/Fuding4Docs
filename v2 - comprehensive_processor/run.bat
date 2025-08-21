
@echo off
rem python main_processor.py --config config_fast.ini
del "C:\Users\dansp\Dropbox\Working Files\GitHub\Funding4Docs\Funding4DocsExploration\v2 - comprehensive_processor\logs\parallel_processor.log"
echo Clearing contents of C:\Funding4Docs_MIRROR...
del "C:\Users\dansp\Dropbox\Working Files\GitHub\Funding4Docs\Funding4DocsExploration\v2 - comprehensive_processor\patient_data.csv"
:: Delete all files in the directory and subdirectories
del /s /q "C:\Funding4Docs_MIRROR\*.*"

:: Delete all subdirectories
for /d %%i in ("C:\Funding4Docs_MIRROR\*") do rmdir /s /q "%%i"

echo Done! C:\Funding4Docs_MIRROR is now empty.

python parallel_main_processor.py  --workers 4