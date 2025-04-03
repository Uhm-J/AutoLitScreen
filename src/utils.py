import os
import sys
import toml
import json
import datetime
from typing import Set, List, Optional, Dict, Tuple, Any
from dataclasses import asdict
from .models import PubMedArticle

def load_config(config_path: str) -> dict:
    """Loads and validates configuration from a TOML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        # --- Validation ---
        core_keys = ["email", "research_question", "ollama_url", "ollama_model", "output_dir", "request_delay_seconds", "ollama_request_timeout"]
        # Add criteria file keys
        criteria_keys = ["inclusion_criteria_file", "exclusion_criteria_file"]
        tiab_keys = ["tiab_prompt_file", "pmid_input_file", "pmid_batch_size"]
        pdf_keys = ["fetch_pdfs", "pdf_download_dir", "unpaywall_email"]
        fts_keys = ["fts_enabled", "fts_prompt_file", "fts_output_dir"]
        mol_keys = ["mol_extraction_enabled", "mol_extraction_prompt_file", "mol_extraction_output_file"]

        # Check core + TIAB + Criteria files as base requirements now
        required_keys = core_keys + tiab_keys + criteria_keys
        missing_keys = [key for key in required_keys if key not in config or config[key] is None or config[key] == ""]
        if missing_keys: raise ValueError(f"Config missing required or empty keys: {missing_keys}")

        # Defaults/validate PDF keys
        config.setdefault('fetch_pdfs', False)
        config.setdefault('pdf_download_dir', os.path.join(config.get('output_dir', 'tiab_screening'), 'pdfs'))
        config.setdefault('unpaywall_email', config.get('email'))
        if config['fetch_pdfs']:
             missing_pdf = [k for k in pdf_keys if k not in config or config[k] is None or config[k] == ""]
             if missing_pdf: raise ValueError(f"Config 'fetch_pdfs' enabled but missing/empty: {missing_pdf}")
             if config["unpaywall_email"] == "your.email@example.com": print(f"WARNING: PDF fetching enabled, please update 'unpaywall_email'.")

        # Defaults/validate FTS keys
        config.setdefault('fts_enabled', False)
        config.setdefault('fts_prompt_file', 'fts_criteria.txt')
        config.setdefault('fts_output_dir', config.get('output_dir'))
        if config['fts_enabled']:
             missing_fts = [k for k in fts_keys if k not in config or config[k] is None or config[k] == ""]
             if missing_fts: raise ValueError(f"Config 'fts_enabled' enabled but missing/empty: {missing_fts}")
             if not config['fetch_pdfs']: print("WARNING: 'fts_enabled' is true but 'fetch_pdfs' is false. FTS requires downloaded PDFs.")
             if not config.get('pdf_download_dir'): raise ValueError("Config 'fts_enabled' enabled but 'pdf_download_dir' is missing.")

        # Defaults/validate Molecule Extraction keys
        config.setdefault('mol_extraction_enabled', False)
        config.setdefault('mol_extraction_prompt_file', 'mol_extract_prompt.txt')
        config.setdefault('mol_extraction_output_file', os.path.join(config.get('output_dir', 'tiab_screening'), 'results.txt'))
        if config['mol_extraction_enabled']:
             missing_mol = [k for k in mol_keys if k not in config or config[k] is None or config[k] == ""]
             if missing_mol: raise ValueError(f"Config 'mol_extraction_enabled' enabled but missing/empty: {missing_mol}")
             # Allow mol extraction even if FTS is off, but warn it usually follows FTS
             if not config['fts_enabled']: print("INFO: 'mol_extraction_enabled' is true but 'fts_enabled' is false. Molecule extraction will run on TIAB accepted PDFs if available.")
             if not config.get('pdf_download_dir'): raise ValueError("Config 'mol_extraction_enabled' enabled but 'pdf_download_dir' is missing.")

        # Validate Entrez email
        if not config["email"] or config["email"] == "your.email@example.com": print(f"WARNING: Please update 'email' in {config_path}.")
        config.setdefault('ollama_request_timeout', 180)

        # Ensure output directories are absolute paths
        config['output_dir'] = os.path.abspath(config['output_dir'])
        config['pdf_download_dir'] = os.path.abspath(config['pdf_download_dir'])
        config['fts_output_dir'] = os.path.abspath(config['fts_output_dir'])
        config['mol_extraction_output_file'] = os.path.abspath(config['mol_extraction_output_file'])

        return config
    except FileNotFoundError: print(f"Error: Config file not found: {config_path}"); sys.exit(1)
    except toml.TomlDecodeError as e: print(f"Error: Could not decode TOML config file {config_path}: {e}"); sys.exit(1)
    except ValueError as e: print(f"Error: Invalid configuration in {config_path}: {e}"); sys.exit(1)
    except Exception as e: print(f"Error loading config {config_path}: {e}"); sys.exit(1)

def create_directory(dir_path: str) -> bool:
    """Creates a directory if it doesn't exist."""
    try: os.makedirs(dir_path, exist_ok=True); return True
    except OSError as e: print(f"Error creating directory {dir_path}: {e}"); return False

def load_pmids_from_file(filepath: str) -> Set[str]:
    """Reads PMIDs (one per line) from a file into a set."""
    pmids = set()
    if not os.path.exists(filepath): return pmids
    try:
        with open(filepath, 'r', encoding='utf-8') as f: pmids = {line.strip() for line in f if line.strip().isdigit()}
    except IOError as e: print(f"Error reading PMID file {filepath}: {e}")
    return pmids

def load_processed_pmids(filepaths: List[str]) -> Set[str]:
    """Loads PMIDs from multiple files into a single set."""
    return {pmid for fp in filepaths for pmid in load_pmids_from_file(fp)}

def write_line_to_file(line: str, filepath: str) -> bool:
    """Appends a line to the specified file."""
    try:
        if not create_directory(os.path.dirname(filepath)): return False
        with open(filepath, 'a', encoding='utf-8') as f: f.write(f"{line}\n")
        return True
    except IOError as e: print(f"  Error writing line to {filepath}: {e}"); return False

def write_pmid_to_file(pmid: str, filepath: str) -> bool:
    """Appends a PMID (as a line) to the specified file."""
    return write_line_to_file(pmid, filepath)

def load_text_file(filepath: str) -> Optional[str]:
     """Loads text content from a file."""
     try:
         with open(filepath, 'r', encoding='utf-8') as f: return f.read()
     except FileNotFoundError: print(f"Error: File not found: {filepath}"); return None
     except IOError as e: print(f"Error reading file {filepath}: {e}"); return None

def save_articles_to_json(articles: Dict[str, PubMedArticle], filepath: str) -> bool:
    """Saves a dictionary of PubMedArticle objects to a JSON file."""
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert dataclasses to dictionaries
        articles_dict = {pmid: asdict(article) for pmid, article in articles.items()}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles_dict, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving articles to {filepath}: {e}")
        return False

def load_articles_from_json(filepath: str) -> Dict[str, PubMedArticle]:
    """Loads PubMedArticle objects from a JSON file."""
    articles: Dict[str, PubMedArticle] = {}
    
    if not os.path.exists(filepath):
        return articles
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            articles_dict = json.load(f)
        
        # Convert dictionaries back to PubMedArticle objects
        for pmid, article_dict in articles_dict.items():
            articles[pmid] = PubMedArticle(**article_dict)
        
        return articles
    except Exception as e:
        print(f"Error loading articles from {filepath}: {e}")
        return {} 