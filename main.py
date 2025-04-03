# -*- coding: utf-8 -*-
"""
Modular Configurable Screener with Title/Abstract (TIAB),
Optional Full-Text Screening (FTS), and Optional Molecule Extraction using
Local Ollama LLM. Research Question from config is prepended to prompts.
Uses TOML for configuration. Loads Inclusion/Exclusion criteria from files.

Includes PDF Fetching via Unpaywall API.

Phase 1 (TIAB): Screens PMIDs based on title/abstract.
Phase 2 (PDF Fetch): Attempts to download PDFs for TIAB-accepted articles.
Phase 3 (FTS): Screens downloaded PDFs based on full text.
Phase 4 (Molecule Extraction): Extracts molecule names from FTS-accepted PDFs.

Requirements:
- Python 3.x
- Biopython library (`pip install biopython`)
- Requests library (`pip install requests`)
- PyMuPDF library (`pip install pymupdf`)
- TOML library (`pip install toml`)
- A locally running Ollama instance (https://ollama.com/)
- An Ollama model pulled (e.g., `ollama pull llama3`)
- Configuration file (`config.toml`)
- TIAB Prompt file (e.g., `tiab_prompt.txt` - with {inclusion}/{exclusion})
- FTS Prompt file (e.g., `fts_prompt.txt` - with {inclusion}/{exclusion})
- Inclusion Criteria file (e.g., `inclusion_criteria.txt`)
- Exclusion Criteria file (e.g., `exclusion_criteria.txt`)
- Molecule Extraction Prompt file (e.g., `mol_extract_prompt.txt`)
- Input PMID file (`pmid_for_review.txt`)

Instructions:
1.  Install libraries: `pip install biopython requests pymupdf toml`
2.  Ensure `config.toml` and all required prompt/criteria/input files exist.
3.  Update `config.toml` with your details, research_question, file paths, and settings.
4.  Ensure `tiab_prompt.txt` and `fts_prompt.txt` contain `{inclusion}` and `{exclusion}`.
5.  Customize criteria files and molecule extraction prompt.
6.  Ensure Ollama is running.
7.  Run the script: `python your_script_name.py`
"""

import os
import sys
import traceback
import requests
import argparse
from datetime import datetime

from src.utils import load_config, load_articles_from_json
from src.screeners import TiabScreener, FTScreener, MoleculeExtractor

def export_to_ris(articles_file, output_file, include_tiab=True, include_fts=True):
    """
    Export articles from JSON to RIS format
    
    Parameters:
    - articles_file: Path to the articles.json file
    - output_file: Path to save the RIS output
    - include_tiab: Include articles accepted in TIAB phase (if FTS not done)
    - include_fts: Include articles accepted in FTS phase if available
    
    Returns:
    - Number of articles exported
    """
    if not os.path.exists(articles_file):
        print(f"Error: Articles file '{articles_file}' not found.")
        return 0
    
    articles = load_articles_from_json(articles_file)
    if not articles:
        print("No articles found in the JSON file.")
        return 0
    
    exported_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for pmid, article in articles.items():
            # Determine if this article should be included
            should_export = False
            
            # If FTS is completed and we want FTS-accepted articles
            if include_fts and article.status_fts == "accepted":
                should_export = True
            # If article passed TIAB but no FTS or we specifically want TIAB articles
            elif include_tiab and article.status_tiab == "accepted":
                should_export = True
                
            if should_export:
                # Use the built-in to_ris method
                ris_content = article.to_ris()
                
                # Add our custom screening status note since it's not in the default to_ris method
                ris_lines = ris_content.split('\n')

                # Write the modified RIS entry
                f.write('\n'.join(ris_lines) + '\n\n')
                exported_count += 1
    
    return exported_count

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Automated Literature Screening")
    parser.add_argument("--ris", action="store_true", help="Export accepted articles to RIS format")
    parser.add_argument("--tiab-only", action="store_true", help="When used with --ris, export only TIAB accepted articles")
    parser.add_argument("--fts-only", action="store_true", help="When used with --ris, export only FTS accepted articles")
    parser.add_argument("--output", "-o", default="accepted_articles.ris", help="Output filename for RIS export (default: accepted_articles.ris)")
    parser.add_argument("--input", "-i", default="pmid_for_review.txt", help="Input filename for PMIDs (default: pmid_for_review.txt)")
    args = parser.parse_args()
    
    CONFIG_FILE = "config.toml"
    if not os.path.exists(CONFIG_FILE): 
        print(f"Error: Config file '{CONFIG_FILE}' not found.")
        sys.exit(1)

    config = {}
    try:
        config = load_config(CONFIG_FILE)
        config['pmid_input_file'] = args.input if args.input else config['pmid_input_file']
        
        # Handle RIS export if requested
        if args.ris:
            articles_file = os.path.join(config['output_dir'], "articles.json")
            include_tiab = not args.fts_only
            include_fts = not args.tiab_only
            
            print(f"\n--- Exporting articles to RIS format ---")
            exported = export_to_ris(
                articles_file, 
                args.output,
                include_tiab=include_tiab,
                include_fts=include_fts
            )
            
            if exported > 0:
                print(f"Successfully exported {exported} articles to {args.output}")
            else:
                print("No articles were exported.")
            
            # Exit after export
            sys.exit(0)
        
        # Check Ollama connection for screening process
        ollama_base_url = config.get("ollama_url", "").replace("/api/generate", "/")
        if not ollama_base_url: raise ValueError("'ollama_url' missing/empty.")
        response = requests.get(ollama_base_url, timeout=5)
        if response.status_code != 200: print(f"Warning: Ollama server at {ollama_base_url} not reachable (status: {response.status_code}).")
        else: print(f"Ollama server found at {ollama_base_url}")
    except requests.exceptions.RequestException as e: print(f"Error: Cannot connect to Ollama server ({CONFIG_FILE}). Details: {e}"); sys.exit(1)
    except Exception as e: print(f"Error during initial setup/Ollama check: {e}"); sys.exit(1)

    # --- Run TIAB Screening Phase ---
    try:
        tiab_screener = TiabScreener(config=config)
        tiab_screener.run()
    except Exception as e: print(f"\n--- Critical error during TIAB screening ---\n{type(e).__name__}: {e}\nTraceback:"); traceback.print_exc(); print("--- Exiting ---"); sys.exit(1)

    # --- Run Full-Text Screening Phase (Optional) ---
    fts_ran_ok = False
    if config.get('fts_enabled', False):
        try:
            fts_screener = FTScreener(config=config)
            fts_screener.run()
            fts_ran_ok = True
        except Exception as e: print(f"\n--- Critical error during Full-Text Screening (FTS) ---\n{type(e).__name__}: {e}\nTraceback:"); traceback.print_exc(); print("--- Exiting ---"); sys.exit(1)
    else:
        print("\nFull-Text Screening (FTS) is disabled in configuration.")

    # --- Run Molecule Extraction Phase (Optional) ---
    if config.get('mol_extraction_enabled', False):
        try:
            # Check if we have articles with the right status
            run_mol_extraction = False
            articles_db = os.path.join(config['output_dir'], config['articles_file'].split(".")[0] + ".json")
            if os.path.exists(articles_db):
                articles = load_articles_from_json(articles_db)
                if config.get('fts_enabled'):
                    # Look for FTS accepted articles
                    fts_accepted_count = sum(1 for article in articles.values() 
                                           if article.status_fts == "accepted")
                    if fts_ran_ok and fts_accepted_count > 0:
                        run_mol_extraction = True
                        print(f"\nFound {fts_accepted_count} FTS accepted articles for molecule extraction.")
                    else:
                        print(f"\nSkipping Molecule Extraction: FTS enabled but failed or no accepted articles found.")
                else:
                    # FTS disabled, look for TIAB accepted articles
                    tiab_accepted_count = sum(1 for article in articles.values() 
                                            if article.status_tiab == "accepted")
                    if tiab_accepted_count > 0:
                        run_mol_extraction = True
                        print(f"\nINFO: FTS disabled. Found {tiab_accepted_count} TIAB accepted articles for molecule extraction.")
                    else:
                        print(f"\nSkipping Molecule Extraction: No TIAB accepted articles found.")
            else:
                print(f"\nSkipping Molecule Extraction: Articles database not found at {articles_db}")
            
            if run_mol_extraction:
                mol_extractor = MoleculeExtractor(config=config)
                mol_extractor.run()
        except Exception as e: print(f"\n--- Critical error during Molecule Extraction ---\n{type(e).__name__}: {e}\nTraceback:"); traceback.print_exc(); print("--- Exiting ---"); sys.exit(1)
    else:
        print("\nMolecule Extraction is disabled in the configuration.")

    print("\n--- All Enabled Phases Complete ---")