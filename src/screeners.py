import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from .utils import (
    create_directory, load_pmids_from_file, load_processed_pmids,
    write_pmid_to_file, load_text_file
)
from .pdf_utils import extract_pdf_text
from .clients import PubMedClient, OllamaClient, PdfDownloader

class TiabScreener:
    """Orchestrates the TIAB screening process."""
    def __init__(self, config: Dict):
        self.config = config; self.research_question = self.config['research_question']
        # Load prompt template
        self.tiab_prompt_template = load_text_file(self.config['tiab_prompt_file'])
        if self.tiab_prompt_template is None: sys.exit(1)
        # Load inclusion/exclusion criteria
        self.inclusion_criteria = load_text_file(self.config['inclusion_criteria_file'])
        self.exclusion_criteria = load_text_file(self.config['exclusion_criteria_file'])
        if self.inclusion_criteria is None or self.exclusion_criteria is None:
            print("Error: Failed to load inclusion or exclusion criteria files specified in config.")
            sys.exit(1)

        self.output_dir = self.config['output_dir']; self.accepted_file = os.path.join(self.output_dir, "accepted.txt"); self.rejected_file = os.path.join(self.output_dir, "rejected.txt"); self.error_file = os.path.join(self.output_dir, "error_screening.txt"); self.pdf_failed_file = os.path.join(self.output_dir, "pdf_download_failed.txt")
        create_directory(self.output_dir); self.processed_pmids = load_processed_pmids([self.accepted_file, self.rejected_file, self.error_file]); print(f"Loaded {len(self.processed_pmids)} previously screened PMIDs (TIAB).")
        self.pubmed_client = PubMedClient(email=self.config['email']); self.ollama_client = OllamaClient(model_name=self.config['ollama_model'], api_url=self.config['ollama_url'], request_timeout=self.config['ollama_request_timeout'])
        self.pdf_downloader: Optional[PdfDownloader] = PdfDownloader(download_dir=self.config['pdf_download_dir'], unpaywall_email=self.config['unpaywall_email']) if self.config.get('fetch_pdfs', False) else None

    def _get_pmids_to_process(self) -> List[str]:
        """Gets PMIDs from input file, excluding processed ones."""
        all_pmids_in_file = load_pmids_from_file(self.config['pmid_input_file'])
        if not all_pmids_in_file: print(f"No PMIDs found in input file: {self.config['pmid_input_file']}"); return []
        pmids_to_process = sorted(list(all_pmids_in_file - self.processed_pmids))
        print(f"Found {len(all_pmids_in_file)} total PMIDs in input file."); print(f"{len(pmids_to_process)} PMIDs remaining to be screened (TIAB).")
        return pmids_to_process

    def _screen_article(self, title: Optional[str], abstract: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Formats TIAB prompt (with RQ and criteria) and uses OllamaClient."""
        effective_title = title if title else "No Title Found"; effective_abstract = abstract if abstract else "No Abstract Found"
        if abstract is None or abstract.strip() == "" or abstract.startswith("No Abstract") or abstract.startswith("Error Fetching"): return None, "Skipped - No Abstract"
        if title is None or title.strip() == "" or title.startswith("No Title") or title.startswith("Error Fetching"): return None, "Skipped - No Title"
        try:
            # Format template with all values at once
            specific_prompt = self.tiab_prompt_template.format(
                inclusion=self.inclusion_criteria,
                exclusion=self.exclusion_criteria,
                title=effective_title,
                abstract=effective_abstract
            )
            # Prepend Research Question
            full_prompt = f"Research Question: {self.research_question}\n\n---\n\n{specific_prompt}"
        except KeyError as e:
             # More specific error if placeholders are missing
             return None, f"Error - Missing placeholder {{{e}}} in TIAB prompt template or criteria files."
        except Exception as e:
             return None, f"Error - Failed to format TIAB prompt: {e}"

        return self.ollama_client.generate(full_prompt)

    def _classify_and_write_result(self, pmid: str, screening_result: Optional[str], error_message: Optional[str]) -> Tuple[str, str]:
        """Classifies TIAB outcome and writes PMID to file."""
        filepath = self.error_file; status = "Error/Unknown"
        if error_message: print(f"  Screening Result: {error_message}"); status = "Error/Skipped"
        elif screening_result:
            print(f"  Ollama Raw Result: {screening_result}")
            outcome = screening_result.lower().strip()
            if outcome.startswith("relevant"): filepath = self.accepted_file; status = "Accepted"
            elif outcome.startswith("not relevant"): filepath = self.rejected_file; status = "Rejected"
            else: print(f"  Warning: Unclear Ollama response for PMID {pmid}. Logging to error file."); status = "Error/Unclear"
        else: print(f"  Warning: No result or error message for PMID {pmid}. Logging to error file.")
        if write_pmid_to_file(pmid, filepath): self.processed_pmids.add(pmid)
        else: status = "Write Error"
        return filepath, status

    def run(self) -> None:
        """Executes the TIAB screening and optional PDF download workflow."""
        print("\n--- Starting TIAB Screening Phase ---")
        pmids_to_process = self._get_pmids_to_process()
        if not pmids_to_process: print("No new PMIDs to screen (TIAB). Exiting TIAB phase."); return
        total_to_process = len(pmids_to_process); run_processed_count = 0; run_accepted_count = 0; run_rejected_count = 0; run_error_count = 0; run_pdf_downloaded_count = 0; run_pdf_failed_count = 0
        batch_size = self.config['pmid_batch_size']; delay = self.config['request_delay_seconds']
        for i in range(0, total_to_process, batch_size):
            batch_pmids = pmids_to_process[i:i + batch_size]
            print(f"\n--- Processing TIAB Batch {i//batch_size + 1}/{(total_to_process + batch_size - 1)//batch_size} (PMIDs {i+1}-{min(i+batch_size, total_to_process)}) ---")
            if i > 0: time.sleep(delay)
            article_details_batch = self.pubmed_client.fetch_details_batch(batch_pmids)
            if not article_details_batch: print(f"Skipping TIAB batch {i//batch_size + 1} due to critical PubMed fetch error."); [self._classify_and_write_result(p, None, "Error - PubMed Batch Fetch Failed") for p in batch_pmids if p not in self.processed_pmids]; run_error_count += len([p for p in batch_pmids if p not in self.processed_pmids]); run_processed_count += len([p for p in batch_pmids if p not in self.processed_pmids]); continue
            for pmid in batch_pmids:
                if pmid not in article_details_batch:
                    if pmid not in self.processed_pmids: _, write_status = self._classify_and_write_result(pmid, None, "Error - Missing in Fetched Batch"); run_error_count += 1; run_processed_count += 1
                    continue
                article = article_details_batch[pmid]; title = article.get('title'); abstract = article.get('abstract'); doi = article.get('doi')
                print(f"\n[{run_processed_count + 1}/{total_to_process}] Screening TIAB PMID: {pmid} (DOI: {doi if doi else 'N/A'})"); print(f"  Title: {str(title)[:100]}{'...' if len(str(title))>100 else ''}")
                time.sleep(delay); screening_result, error_message = self._screen_article(title, abstract); filepath, write_status = self._classify_and_write_result(pmid, screening_result, error_message); print(f"  Status: {write_status}")
                if write_status == "Accepted": run_accepted_count += 1
                elif write_status == "Rejected": run_rejected_count += 1
                elif write_status.startswith("Error") or write_status == "Write Error": run_error_count += 1
                run_processed_count += 1
                if write_status == "Accepted" and self.pdf_downloader:
                     pdf_success = self.pdf_downloader.download_pdf(pmid, doi, title)
                     if pdf_success: run_pdf_downloaded_count += 1
                     else: run_pdf_failed_count += 1; print(f"  PDF Download Failed: Logging PMID {pmid} to {os.path.basename(self.pdf_failed_file)}"); write_pmid_to_file(pmid, self.pdf_failed_file)
                     time.sleep(delay)
        print("\n--- TIAB Screening Phase Complete ---")
        final_accepted = len(load_pmids_from_file(self.accepted_file)); final_rejected = len(load_pmids_from_file(self.rejected_file)); final_errors = len(load_pmids_from_file(self.error_file)); final_pdf_failed = len(load_pmids_from_file(self.pdf_failed_file)); final_processed_total = final_accepted + final_rejected + final_errors
        print(f"\n--- TIAB Run Summary ---"); print(f"Total PMIDs processed/attempted screening in this run: {run_processed_count}"); print(f"  Accepted: {run_accepted_count}, Rejected: {run_rejected_count}, Errors/Skipped: {run_error_count}")
        if self.config.get('fetch_pdfs', False): print(f"  PDFs downloaded: {run_pdf_downloaded_count}, Failed/Skipped PDF downloads: {run_pdf_failed_count}")
        print("\n--- TIAB Final Counts ---"); print(f"Total Accepted: {final_accepted}, Rejected: {final_rejected}, Errors/Skipped: {final_errors}, Screened: {final_processed_total}")
        if self.config.get('fetch_pdfs', False): print(f"Total PDFs logged as Failed Download: {final_pdf_failed}")
        print(f"\nTIAB results saved in '{self.output_dir}'.")
        if self.config.get('fetch_pdfs', False): print(f"PDFs saved in '{self.config['pdf_download_dir']}'. Failed PDF PMIDs logged to '{os.path.basename(self.pdf_failed_file)}'.")
        print("---------------------------------------")

class FTScreener:
    """Orchestrates the Full-Text Screening (FTS) process."""
    def __init__(self, config: Dict):
        self.config = config; self.research_question = self.config['research_question']
        if not self.config.get('fts_enabled'): raise ValueError("FTScreener initialized but fts_enabled is false.")
        # Load prompt template
        self.fts_prompt_template = load_text_file(self.config['fts_prompt_file'])
        if self.fts_prompt_template is None: sys.exit(1)
        # Load inclusion/exclusion criteria
        self.inclusion_criteria = load_text_file(self.config['inclusion_criteria_file'])
        self.exclusion_criteria = load_text_file(self.config['exclusion_criteria_file'])
        if self.inclusion_criteria is None or self.exclusion_criteria is None:
            print("Error: Failed to load inclusion or exclusion criteria files specified in config.")
            sys.exit(1)

        self.fts_output_dir = self.config['fts_output_dir']; self.pdf_download_dir = self.config['pdf_download_dir']; self.tiab_accepted_file = os.path.join(self.config['output_dir'], "accepted.txt")
        self.fts_accepted_file = os.path.join(self.fts_output_dir, "fts_accepted.txt"); self.fts_rejected_file = os.path.join(self.fts_output_dir, "fts_rejected.txt"); self.fts_error_file = os.path.join(self.fts_output_dir, "fts_error.txt")
        create_directory(self.fts_output_dir); self.fts_processed_pmids = load_processed_pmids([self.fts_accepted_file, self.fts_rejected_file, self.fts_error_file]); print(f"Loaded {len(self.fts_processed_pmids)} previously processed PMIDs (FTS).")
        self.ollama_client = OllamaClient(model_name=self.config['ollama_model'], api_url=self.config['ollama_url'], request_timeout=self.config['ollama_request_timeout']); self.delay = self.config['request_delay_seconds']

    def _get_pmids_for_fts(self) -> List[str]:
        """Gets PMIDs from TIAB accepted list, excluding FTS processed."""
        tiab_accepted_pmids = load_pmids_from_file(self.tiab_accepted_file)
        if not tiab_accepted_pmids: print(f"No PMIDs found in TIAB accepted file: {self.tiab_accepted_file}. Cannot perform FTS."); return []
        pmids_to_process = sorted(list(tiab_accepted_pmids - self.fts_processed_pmids))
        print(f"Found {len(tiab_accepted_pmids)} total PMIDs in TIAB accepted file."); print(f"{len(pmids_to_process)} PMIDs remaining for Full-Text Screening (FTS).")
        return pmids_to_process

    def _screen_full_text(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Formats FTS prompt (with RQ and criteria) and uses OllamaClient."""
        if not full_text and full_text != "": # Allow empty string if PDF extraction yielded empty
             return None, "Error - Invalid text provided for FTS"
        print(f"  Sending extracted text (approx {len(full_text)} chars) to Ollama for FTS...")
        try:
            # Format template with criteria first
            prompt_with_criteria = self.fts_prompt_template.format(
                inclusion=self.inclusion_criteria,
                exclusion=self.exclusion_criteria,
                full_text=full_text # Also format full_text here if placeholder is {full_text}
            )
            # Prepend Research Question
            full_prompt = f"Research Question: {self.research_question}\n\n---\n\n{prompt_with_criteria}"
        except KeyError as e:
             # More specific error if placeholders are missing
             return None, f"Error - Missing placeholder {{{e}}} in FTS prompt template or criteria files."
        except Exception as e:
             return None, f"Error - Failed to format FTS prompt: {e}"

        return self.ollama_client.generate(full_prompt)

    def _classify_and_write_fts_result(self, pmid: str, screening_result: Optional[str], error_message: Optional[str]) -> str:
        """Classifies FTS outcome and writes PMID to FTS result files."""
        filepath = self.fts_error_file; status = "Error/Unknown"
        if error_message: print(f"  FTS Screening Result: {error_message}"); status = "Error/Skipped"
        elif screening_result:
            print(f"  Ollama FTS Raw Result: {screening_result}")
            outcome = screening_result.lower().strip()
            if outcome.startswith("fts relevant"): filepath = self.fts_accepted_file; status = "FTS Accepted"
            elif outcome.startswith("fts not relevant"): filepath = self.fts_rejected_file; status = "FTS Rejected"
            else: print(f"  Warning: Unclear Ollama FTS response for PMID {pmid}. Logging to FTS error file."); status = "Error/Unclear FTS"
        else: print(f"  Warning: No FTS result or error message for PMID {pmid}. Logging to FTS error file.")
        if write_pmid_to_file(pmid, filepath): self.fts_processed_pmids.add(pmid)
        else: status = "Write Error FTS"
        return status

    def run(self) -> None:
        """Executes the Full-Text Screening workflow."""
        print("\n--- Starting Full-Text Screening (FTS) Phase ---")
        if not self.config.get('fts_enabled'): print("FTS is disabled in configuration. Skipping."); return
        if not os.path.exists(self.pdf_download_dir): print(f"Error: PDF download directory not found: {self.pdf_download_dir}. Cannot perform FTS."); return
        pmids_for_fts = self._get_pmids_for_fts()
        if not pmids_for_fts: print("No new PMIDs found for FTS processing."); return
        total_to_process = len(pmids_for_fts); run_processed_count = 0; run_fts_accepted_count = 0; run_fts_rejected_count = 0; run_fts_error_count = 0
        for i, pmid in enumerate(pmids_for_fts):
            print(f"\n[{i + 1}/{total_to_process}] Processing FTS for PMID: {pmid}")
            pdf_filename = f"{pmid}.pdf"; pdf_path = os.path.join(self.pdf_download_dir, pdf_filename)
            full_text = extract_pdf_text(pdf_path)
            # Handle case where PDF exists but text extraction fails (returns None)
            # Also handle case where extraction works but text is empty (returns "")
            if full_text is None:
                 status = self._classify_and_write_fts_result(pmid, None, f"Error - PDF Text Extraction Failed ({pdf_filename})")
                 print(f"  Status: {status}"); run_fts_error_count += 1; run_processed_count += 1; time.sleep(self.delay); continue
            elif full_text == "":
                 status = self._classify_and_write_fts_result(pmid, None, f"Error - PDF Text Extraction Empty ({pdf_filename})")
                 print(f"  Status: {status}"); run_fts_error_count += 1; run_processed_count += 1; time.sleep(self.delay); continue

            # Proceed if text extraction succeeded (even if empty string was returned initially, now handled)
            time.sleep(self.delay); screening_result, error_message = self._screen_full_text(full_text); status = self._classify_and_write_fts_result(pmid, screening_result, error_message); print(f"  Status: {status}")
            if status == "FTS Accepted": run_fts_accepted_count += 1
            elif status == "FTS Rejected": run_fts_rejected_count += 1
            elif status.startswith("Error") or status == "Write Error FTS": run_fts_error_count += 1
            run_processed_count += 1
        print("\n--- Full-Text Screening (FTS) Phase Complete ---")
        final_fts_accepted = len(load_pmids_from_file(self.fts_accepted_file)); final_fts_rejected = len(load_pmids_from_file(self.fts_rejected_file)); final_fts_errors = len(load_pmids_from_file(self.fts_error_file)); final_fts_processed_total = final_fts_accepted + final_fts_rejected + final_fts_errors
        print(f"\n--- FTS Run Summary ---"); print(f"Total PMIDs processed/attempted in this FTS run: {run_processed_count}"); print(f"  FTS Accepted: {run_fts_accepted_count}, Rejected: {run_fts_rejected_count}, Errors/Skipped: {run_fts_error_count}")
        print("\n--- FTS Final Counts ---"); print(f"Total FTS Accepted: {final_fts_accepted}, Rejected: {final_fts_rejected}, Errors/Skipped: {final_fts_errors}, Processed: {final_fts_processed_total}")
        print(f"\nFTS results saved in '{self.fts_output_dir}'.")
        print("--------------------------------------------")

class MoleculeExtractor:
    """Extracts molecule names from full texts of accepted articles."""
    def __init__(self, config: Dict):
        self.config = config; self.research_question = self.config['research_question']
        if not self.config.get('mol_extraction_enabled'): raise ValueError("MoleculeExtractor initialized but mol_extraction_enabled is false.")
        self.mol_prompt_template = load_text_file(self.config['mol_extraction_prompt_file'])
        if self.mol_prompt_template is None: sys.exit(1)
        self.pdf_download_dir = self.config['pdf_download_dir'];
        # Determine source file based on FTS status
        self.source_pmid_file = os.path.join(self.config['fts_output_dir'], "fts_accepted.txt") if self.config.get('fts_enabled') else os.path.join(self.config['output_dir'], "accepted.txt")
        self.output_file = self.config['mol_extraction_output_file']
        if create_directory(os.path.dirname(self.output_file)):
             if os.path.exists(self.output_file):
                  try: os.remove(self.output_file); print(f"Cleared previous molecule extraction results file: {self.output_file}")
                  except OSError as e: print(f"Warning: Could not clear previous results file {self.output_file}: {e}")
        self.ollama_client = OllamaClient(model_name=self.config['ollama_model'], api_url=self.config['ollama_url'], request_timeout=self.config['ollama_request_timeout']); self.delay = self.config['request_delay_seconds']

    def _get_pmids_for_mol_extraction(self) -> List[str]:
        """Gets PMIDs from FTS or TIAB accepted list."""
        source_desc = "FTS accepted" if self.config.get('fts_enabled') else "TIAB accepted"
        accepted_pmids = load_pmids_from_file(self.source_pmid_file)
        if not accepted_pmids: print(f"No PMIDs found in {source_desc} file: {self.source_pmid_file}. Cannot perform Molecule Extraction."); return []
        print(f"Found {len(accepted_pmids)} PMIDs in {source_desc} file for Molecule Extraction.")
        return sorted(list(accepted_pmids))

    def _extract_molecules_from_text(self, full_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """Formats molecule prompt (with RQ), uses Ollama, and parses the response."""
        if not full_text and full_text != "": return None, "Error - Empty text provided for molecule extraction"
        print(f"  Sending extracted text (approx {len(full_text)} chars) to Ollama for Molecule Extraction...")
        try:
            # Prepend Research Question to the specific prompt template
            specific_prompt = self.mol_prompt_template.format(full_text=full_text)
            full_prompt = f"Research Question: {self.research_question}\n\n---\n\n{specific_prompt}"
        except KeyError as e:
             # Handle case where mol prompt might not have {full_text} if modified, though unlikely
             return None, f"Error - Missing placeholder {{{e}}} in molecule extraction prompt template."
        except Exception as e:
             return None, f"Error - Failed to format molecule extraction prompt: {e}"

        response_text, error_message = self.ollama_client.generate(full_prompt)
        if error_message: return None, error_message
        if response_text:
            molecules = [line.strip() for line in response_text.splitlines() if line.strip() and line.strip().lower() != "none"]
            if molecules: print(f"  Extracted Molecules: {', '.join(molecules)}"); return molecules, None
            else: print("  No specific molecules identified by LLM."); return [], None
        else: return None, "Error - Empty response received from Ollama during molecule extraction"

    def _write_results(self, pmid: str, molecules: List[str]) -> int:
        """Writes 'PMID: molecule' lines to the output file."""
        count = 0
        if not molecules: return count
        for molecule in molecules: line = f"{pmid}: {molecule}"; count += 1 if write_line_to_file(line, self.output_file) else 0
        return count

    def run(self) -> None:
        """Executes the Molecule Extraction workflow."""
        print("\n--- Starting Molecule Extraction Phase ---")
        if not self.config.get('mol_extraction_enabled'): print("Molecule Extraction is disabled. Skipping."); return
        if not os.path.exists(self.pdf_download_dir): print(f"Error: PDF directory not found: {self.pdf_download_dir}. Cannot perform Molecule Extraction."); return
        pmids_to_process = self._get_pmids_for_mol_extraction()
        if not pmids_to_process: print("No PMIDs found for Molecule Extraction processing."); return
        total_to_process = len(pmids_to_process); run_processed_count = 0; run_extraction_errors = 0; total_molecules_extracted = 0
        for i, pmid in enumerate(pmids_to_process):
            print(f"\n[{i + 1}/{total_to_process}] Processing Molecule Extraction for PMID: {pmid}")
            pdf_filename = f"{pmid}.pdf"; pdf_path = os.path.join(self.pdf_download_dir, pdf_filename)
            full_text = extract_pdf_text(pdf_path)
            if full_text is None: print(f"  Skipping PMID {pmid} due to PDF text extraction failure."); run_extraction_errors += 1; run_processed_count += 1; time.sleep(self.delay); continue
            # Handle empty text after successful extraction
            elif full_text == "": print(f"  Skipping PMID {pmid} due to empty extracted text."); run_extraction_errors += 1; run_processed_count += 1; time.sleep(self.delay); continue

            time.sleep(self.delay); molecules, error_message = self._extract_molecules_from_text(full_text)
            if error_message: print(f"  Skipping PMID {pmid} due to Ollama error: {error_message}"); run_extraction_errors += 1
            elif molecules is not None: written_count = self._write_results(pmid, molecules); total_molecules_extracted += written_count
            else: print(f"  Skipping PMID {pmid} due to unknown molecule extraction error."); run_extraction_errors +=1
            run_processed_count += 1
        print("\n--- Molecule Extraction Phase Complete ---")
        print(f"\n--- Molecule Extraction Run Summary ---"); print(f"Total PMIDs processed/attempted: {run_processed_count}"); print(f"Total 'PMID: molecule' lines written: {total_molecules_extracted}"); print(f"PMIDs skipped due to errors: {run_extraction_errors}")
        print(f"\nMolecule extraction results saved to '{self.output_file}'.")
        print("-----------------------------------------") 