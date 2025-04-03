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

from src.utils import load_config, load_articles_from_json
from src.screeners import TiabScreener, FTScreener, MoleculeExtractor

if __name__ == "__main__":
    CONFIG_FILE = "config.toml"
    if not os.path.exists(CONFIG_FILE): print(f"Error: Config file '{CONFIG_FILE}' not found."); sys.exit(1)

    config = {}
    try:
        config = load_config(CONFIG_FILE)
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