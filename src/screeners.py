import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from .utils import (
    create_directory, load_pmids_from_file, load_processed_pmids,
    write_pmid_to_file, load_text_file, save_articles_to_json, load_articles_from_json
)
from .pdf_utils import extract_pdf_text
from .clients import PubMedClient, OllamaClient, PdfDownloader
from .models import PubMedArticle

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

        self.output_dir = self.config['output_dir']
        self.articles_file = os.path.join(self.output_dir, "articles.json")
        create_directory(self.output_dir)
        
        # Load existing articles if the file exists
        self.articles = load_articles_from_json(self.articles_file)
        print(f"Loaded {len(self.articles)} previously processed articles.")
        
        # For backward compatibility, still check if files exist and maintain them
        self.accepted_file = os.path.join(self.output_dir, "accepted.txt")
        self.rejected_file = os.path.join(self.output_dir, "rejected.txt")
        self.error_file = os.path.join(self.output_dir, "error_screening.txt")
        self.pdf_failed_file = os.path.join(self.output_dir, "pdf_download_failed.txt")
        
        # Initialize clients
        self.pubmed_client = PubMedClient(email=self.config['email'])
        self.ollama_client = OllamaClient(model_name=self.config['ollama_model'], api_url=self.config['ollama_url'], request_timeout=self.config['ollama_request_timeout'])
        self.pdf_downloader = PdfDownloader(download_dir=self.config['pdf_download_dir'], unpaywall_email=self.config['unpaywall_email']) if self.config.get('fetch_pdfs', False) else None

    def _get_pmids_to_process(self) -> List[str]:
        """Gets PMIDs from input file, excluding processed ones."""
        all_pmids_in_file = load_pmids_from_file(self.config['pmid_input_file'])
        if not all_pmids_in_file: print(f"No PMIDs found in input file: {self.config['pmid_input_file']}"); return []
        
        # Find PMIDs that haven't been processed or had errors
        pmids_to_process = [pmid for pmid in all_pmids_in_file if 
                           pmid not in self.articles or 
                           self.articles[pmid].status_tiab in ["error", "failed"]]
        
        pmids_to_process = sorted(pmids_to_process)
        print(f"Found {len(all_pmids_in_file)} total PMIDs in input file.")
        print(f"{len(pmids_to_process)} PMIDs remaining to be screened (TIAB).")
        return pmids_to_process

    def _screen_article(self, article: PubMedArticle) -> Tuple[Optional[str], Optional[str]]:
        """Formats TIAB prompt (with RQ and criteria) and uses OllamaClient."""
        title = article.title
        abstract = article.abstract
        
        if abstract is None or abstract.strip() == "" or abstract.startswith("No Abstract") or abstract.startswith("Error"): 
            return None, "Skipped - No Abstract"
        if title is None or title.strip() == "" or title.startswith("No Title") or title.startswith("Error"): 
            return None, "Skipped - No Title"
        
        try:
            # Format template with criteria first
            prompt_with_criteria = self.tiab_prompt_template.format(
                inclusion=self.inclusion_criteria,
                exclusion=self.exclusion_criteria,
                title=title,
                abstract=abstract
            )
            # Prepend Research Question to prompt template
            full_prompt = f"Research Question: {self.research_question}\n\n---\n\n{prompt_with_criteria}"
        except KeyError as e:
            return None, f"Error - Missing placeholder {{{e}}} in TIAB prompt template or criteria files."
        except Exception as e:
            return None, f"Error - Failed to format TIAB prompt: {e}"

        # Submit to Ollama for screening
        return self.ollama_client.generate(full_prompt)

    def _classify_and_update_article(self, article: PubMedArticle, screening_result: Optional[str], error_message: Optional[str]) -> str:
        """Updates article object with screening results and returns status string."""
        status = "Unknown"
        
        if error_message:
            article.status_tiab = "error"
            status = f"Error - {error_message}"
            # For legacy compatibility
            write_pmid_to_file(article.pmid, self.error_file)
        elif screening_result:
            outcome = screening_result.lower().strip()
            if "relevant" in outcome:
                article.status_tiab = "accepted"
                status = "Accepted"
                # For legacy compatibility
                write_pmid_to_file(article.pmid, self.accepted_file)
            else:
                article.status_tiab = "rejected"
                status = "Rejected"
                # For legacy compatibility
                write_pmid_to_file(article.pmid, self.rejected_file)
        else:
            article.status_tiab = "error"
            status = "Error - No result"
            # For legacy compatibility
            write_pmid_to_file(article.pmid, self.error_file)
            
        # Save updated article data
        self.articles[article.pmid] = article
        save_articles_to_json(self.articles, self.articles_file)
        
        return status

    def run(self) -> None:
        """Executes the TIAB screening and optional PDF download workflow."""
        print("\n--- Starting TIAB Screening Phase ---")
        pmids_to_process = self._get_pmids_to_process()
        if not pmids_to_process: print("No new PMIDs to screen (TIAB). Exiting TIAB phase."); return
        
        total_to_process = len(pmids_to_process)
        run_processed_count = 0
        run_accepted_count = 0
        run_rejected_count = 0
        run_error_count = 0
        run_pdf_downloaded_count = 0
        run_pdf_failed_count = 0
        
        batch_size = self.config['pmid_batch_size']
        delay = self.config['request_delay_seconds']
        
        for i in range(0, total_to_process, batch_size):
            batch_pmids = pmids_to_process[i:i + batch_size]
            print(f"\n--- Processing TIAB Batch {i//batch_size + 1}/{(total_to_process + batch_size - 1)//batch_size} (PMIDs {i+1}-{min(i+batch_size, total_to_process)}) ---")
            
            if i > 0:
                time.sleep(delay)
                
            article_batch = self.pubmed_client.fetch_details_batch(batch_pmids)
            
            if not article_batch:
                print(f"Skipping TIAB batch {i//batch_size + 1} due to critical PubMed fetch error.")
                # Create error articles for each PMID in the batch
                for pmid in batch_pmids:
                    if pmid not in self.articles:
                        error_article = PubMedArticle(
                            title="Error - PubMed Batch Fetch Failed",
                            authors=[],
                            journal="",
                            year=0,
                            pmid=pmid,
                            status_tiab="error"
                        )
                        self.articles[pmid] = error_article
                        # For legacy compatibility
                        write_pmid_to_file(pmid, self.error_file)
                        run_error_count += 1
                        run_processed_count += 1
                
                save_articles_to_json(self.articles, self.articles_file)
                continue
                
            for pmid in batch_pmids:
                if pmid not in article_batch:
                    if pmid not in self.articles:
                        error_article = PubMedArticle(
                            title="Error - Missing in Fetched Batch",
                            authors=[],
                            journal="",
                            year=0,
                            pmid=pmid,
                            status_tiab="error"
                        )
                        self.articles[pmid] = error_article
                        # For legacy compatibility
                        write_pmid_to_file(pmid, self.error_file)
                        run_error_count += 1
                        run_processed_count += 1
                    continue
                
                article = article_batch[pmid]
                print(f"\n[{run_processed_count + 1}/{total_to_process}] Screening TIAB PMID: {pmid} (DOI: {article.doi if article.doi else 'N/A'})")
                print(f"  Title: {str(article.title)[:100]}{'...' if len(str(article.title))>100 else ''}")
                
                time.sleep(delay)
                screening_result, error_message = self._screen_article(article)
                status = self._classify_and_update_article(article, screening_result, error_message)
                print(f"  Status: {status}")
                
                if status == "Accepted":
                    run_accepted_count += 1
                elif status == "Rejected":
                    run_rejected_count += 1
                elif status.startswith("Error"):
                    run_error_count += 1
                    
                run_processed_count += 1
                
                # Download PDF if article was accepted
                if status == "Accepted" and self.pdf_downloader:
                    pdf_success = self.pdf_downloader.download_pdf(article)
                    if pdf_success:
                        run_pdf_downloaded_count += 1
                    else:
                        run_pdf_failed_count += 1
                        print(f"  PDF Download Failed: Logging PMID {pmid} to {os.path.basename(self.pdf_failed_file)}")
                        write_pmid_to_file(pmid, self.pdf_failed_file)
                    
                    # Update article in the dictionary
                    self.articles[pmid] = article
                    save_articles_to_json(self.articles, self.articles_file)
                    
                    time.sleep(delay)
        
        print("\n--- TIAB Screening Phase Complete ---")
        
        # Count articles by status for final report
        final_accepted = sum(1 for article in self.articles.values() if article.status_tiab == "accepted")
        final_rejected = sum(1 for article in self.articles.values() if article.status_tiab == "rejected")
        final_errors = sum(1 for article in self.articles.values() if article.status_tiab in ["error", "failed"])
        final_pdf_failed = sum(1 for article in self.articles.values() if article.status_pdf_fetch in ["error", "failed"])
        final_processed_total = final_accepted + final_rejected + final_errors
        
        print(f"\n--- TIAB Run Summary ---")
        print(f"Total PMIDs processed/attempted screening in this run: {run_processed_count}")
        print(f"  Accepted: {run_accepted_count}, Rejected: {run_rejected_count}, Errors/Skipped: {run_error_count}")
        
        if self.config.get('fetch_pdfs', False):
            print(f"  PDFs downloaded: {run_pdf_downloaded_count}, Failed/Skipped PDF downloads: {run_pdf_failed_count}")
        
        print("\n--- TIAB Final Counts ---")
        print(f"Total Accepted: {final_accepted}, Rejected: {final_rejected}, Errors/Skipped: {final_errors}, Screened: {final_processed_total}")
        
        if self.config.get('fetch_pdfs', False):
            print(f"Total PDFs logged as Failed Download: {final_pdf_failed}")
        
        print(f"\nTIAB results saved in '{self.output_dir}'.")
        
        if self.config.get('fetch_pdfs', False):
            print(f"PDFs saved in '{self.config['pdf_download_dir']}'. Failed PDF PMIDs logged to '{os.path.basename(self.pdf_failed_file)}'.")
        
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

        self.output_dir = self.config['output_dir']
        self.fts_output_dir = self.config['fts_output_dir']
        self.pdf_download_dir = self.config['pdf_download_dir']
        
        # Load articles data from TIAB phase
        self.articles_file = os.path.join(self.output_dir, "articles.json")
        self.articles = load_articles_from_json(self.articles_file)
        print(f"Loaded {len(self.articles)} articles from TIAB phase.")
        
        # For backward compatibility
        self.tiab_accepted_file = os.path.join(self.config['output_dir'], "accepted.txt")
        self.fts_accepted_file = os.path.join(self.fts_output_dir, "fts_accepted.txt")
        self.fts_rejected_file = os.path.join(self.fts_output_dir, "fts_rejected.txt")
        self.fts_error_file = os.path.join(self.fts_output_dir, "fts_error.txt")
        
        create_directory(self.fts_output_dir)
        
        self.ollama_client = OllamaClient(
            model_name=self.config['ollama_model'], 
            api_url=self.config['ollama_url'], 
            request_timeout=self.config['ollama_request_timeout']
        )
        self.delay = self.config['request_delay_seconds']

    def _get_articles_for_fts(self) -> List[str]:
        """Gets PMIDs for FTS from TIAB accepted articles that haven't had FTS yet."""
        tiab_accepted_pmids = set()
        
        # Get articles that were accepted in TIAB phase and need FTS
        for pmid, article in self.articles.items():
            if article.status_tiab == "accepted" and article.status_fts == "pending":
                if article.pdf_path and os.path.exists(article.pdf_path):
                    tiab_accepted_pmids.add(pmid)
                else:
                    # Try the default PDF name pattern
                    pdf_path = os.path.join(self.pdf_download_dir, f"{pmid}.pdf")
                    if os.path.exists(pdf_path):
                        article.pdf_path = pdf_path
                        tiab_accepted_pmids.add(pmid)
        
        if not tiab_accepted_pmids:
            print(f"No new articles found for Full-Text Screening.")
            return []
        
        print(f"Found {len(tiab_accepted_pmids)} articles for Full-Text Screening (FTS).")
        return sorted(list(tiab_accepted_pmids))

    def _screen_full_text(self, article: PubMedArticle) -> Tuple[Optional[str], Optional[str]]:
        """Formats FTS prompt (with RQ and criteria) and uses OllamaClient."""
        if not article.pdf_path or not os.path.exists(article.pdf_path):
            return None, "Error - PDF file not found"
            
        full_text = extract_pdf_text(article.pdf_path)
        
        if full_text is None:
            return None, f"Error - PDF Text Extraction Failed ({os.path.basename(article.pdf_path)})"
        elif full_text == "":
            return None, f"Error - PDF Text Extraction Empty ({os.path.basename(article.pdf_path)})"
            
        print(f"  Sending extracted text (approx {len(full_text)} chars) to Ollama for FTS...")
        try:
            # Format template with criteria and full_text
            prompt_with_criteria = self.fts_prompt_template.format(
                inclusion=self.inclusion_criteria,
                exclusion=self.exclusion_criteria,
                full_text=full_text
            )
            # Prepend Research Question
            full_prompt = f"Research Question: {self.research_question}\n\n---\n\n{prompt_with_criteria}"
        except KeyError as e:
            return None, f"Error - Missing placeholder {{{e}}} in FTS prompt template or criteria files."
        except Exception as e:
            return None, f"Error - Failed to format FTS prompt: {e}"

        return self.ollama_client.generate(full_prompt)

    def _update_article_fts_result(self, article: PubMedArticle, screening_result: Optional[str], error_message: Optional[str]) -> str:
        """Updates article with FTS results and returns status string."""
        status = "Unknown"
        
        if error_message:
            article.status_fts = "error"
            status = f"Error - {error_message}"
            # For legacy compatibility
            write_pmid_to_file(article.pmid, self.fts_error_file)
        elif screening_result:
            print(f"  Ollama FTS Raw Result: {screening_result}")
            outcome = screening_result.lower().strip()
            if "fts relevant" in outcome:
                article.status_fts = "accepted"
                status = "FTS Accepted"
                # For legacy compatibility
                write_pmid_to_file(article.pmid, self.fts_accepted_file)
            elif "fts not relevant" in outcome:
                article.status_fts = "rejected"
                status = "FTS Rejected"
                # For legacy compatibility
                write_pmid_to_file(article.pmid, self.fts_rejected_file)
            else:
                article.status_fts = "error"
                status = "Error - Unclear FTS"
                # For legacy compatibility
                write_pmid_to_file(article.pmid, self.fts_error_file)
        else:
            article.status_fts = "error"
            status = "Error - No FTS Result"
            # For legacy compatibility
            write_pmid_to_file(article.pmid, self.fts_error_file)
            
        # Save updated articles data
        self.articles[article.pmid] = article
        save_articles_to_json(self.articles, self.articles_file)
        
        return status

    def run(self) -> None:
        """Executes the Full-Text Screening workflow."""
        print("\n--- Starting Full-Text Screening (FTS) Phase ---")
        if not self.config.get('fts_enabled'):
            print("FTS is disabled in configuration. Skipping.")
            return
            
        if not os.path.exists(self.pdf_download_dir):
            print(f"Error: PDF download directory not found: {self.pdf_download_dir}. Cannot perform FTS.")
            return
            
        pmids_for_fts = self._get_articles_for_fts()
        if not pmids_for_fts:
            print("No new articles found for FTS processing.")
            return
            
        total_to_process = len(pmids_for_fts)
        run_processed_count = 0
        run_fts_accepted_count = 0
        run_fts_rejected_count = 0
        run_fts_error_count = 0
        
        for i, pmid in enumerate(pmids_for_fts):
            article = self.articles[pmid]
            print(f"\n[{i + 1}/{total_to_process}] Processing FTS for PMID: {pmid}")
            print(f"  Title: {article.title[:100]}{'...' if len(article.title) > 100 else ''}")
            
            time.sleep(self.delay)
            screening_result, error_message = self._screen_full_text(article)
            status = self._update_article_fts_result(article, screening_result, error_message)
            print(f"  Status: {status}")
            
            if status == "FTS Accepted":
                run_fts_accepted_count += 1
            elif status == "FTS Rejected":
                run_fts_rejected_count += 1
            elif status.startswith("Error"):
                run_fts_error_count += 1
                
            run_processed_count += 1
            
        print("\n--- Full-Text Screening (FTS) Phase Complete ---")
        
        # Count articles by status for final report
        final_fts_accepted = sum(1 for article in self.articles.values() if article.status_fts == "accepted")
        final_fts_rejected = sum(1 for article in self.articles.values() if article.status_fts == "rejected")
        final_fts_errors = sum(1 for article in self.articles.values() if article.status_fts in ["error", "failed"])
        final_fts_processed_total = final_fts_accepted + final_fts_rejected + final_fts_errors
        
        print(f"\n--- FTS Run Summary ---")
        print(f"Total articles processed/attempted in this FTS run: {run_processed_count}")
        print(f"  FTS Accepted: {run_fts_accepted_count}, Rejected: {run_fts_rejected_count}, Errors/Skipped: {run_fts_error_count}")
        
        print("\n--- FTS Final Counts ---")
        print(f"Total FTS Accepted: {final_fts_accepted}, Rejected: {final_fts_rejected}, Errors/Skipped: {final_fts_errors}, Processed: {final_fts_processed_total}")
        
        print(f"\nFTS results saved in '{self.fts_output_dir}'.")
        print("--------------------------------------------")

class MoleculeExtractor:
    """Extracts molecule names from full texts of accepted articles."""
    def __init__(self, config: Dict):
        self.config = config; self.research_question = self.config['research_question']
        if not self.config.get('mol_extraction_enabled'): raise ValueError("MoleculeExtractor initialized but mol_extraction_enabled is false.")
        self.mol_prompt_template = load_text_file(self.config['mol_extraction_prompt_file'])
        if self.mol_prompt_template is None: sys.exit(1)
        
        self.output_dir = self.config['output_dir']
        self.pdf_download_dir = self.config['pdf_download_dir']
        
        # Load articles data from previous phases
        self.articles_file = os.path.join(self.output_dir, "articles.json")
        self.articles = load_articles_from_json(self.articles_file)
        print(f"Loaded {len(self.articles)} articles from previous phases.")
        
        # For backward compatibility and results file
        self.output_file = self.config['mol_extraction_output_file']
        create_directory(os.path.dirname(self.output_file))
        
        if os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
                print(f"Cleared previous molecule extraction results file: {self.output_file}")
            except OSError as e:
                print(f"Warning: Could not clear previous results file {self.output_file}: {e}")
                
        self.ollama_client = OllamaClient(
            model_name=self.config['ollama_model'], 
            api_url=self.config['ollama_url'], 
            request_timeout=self.config['ollama_request_timeout']
        )
        self.delay = self.config['request_delay_seconds']

    def _get_articles_for_mol_extraction(self) -> List[str]:
        """Gets PMIDs for molecule extraction from accepted articles."""
        candidate_pmids = []
        
        # Use FTS accepted articles if FTS was enabled, otherwise use TIAB accepted
        for pmid, article in self.articles.items():
            if self.config.get('fts_enabled') and article.status_fts == "accepted":
                if article.pdf_path and os.path.exists(article.pdf_path):
                    candidate_pmids.append(pmid)
                else:
                    # Try standard file naming pattern
                    pdf_path = os.path.join(self.pdf_download_dir, f"{pmid}.pdf")
                    if os.path.exists(pdf_path):
                        article.pdf_path = pdf_path
                        candidate_pmids.append(pmid)
            elif not self.config.get('fts_enabled') and article.status_tiab == "accepted":
                if article.pdf_path and os.path.exists(article.pdf_path):
                    candidate_pmids.append(pmid)
                else:
                    # Try standard file naming pattern
                    pdf_path = os.path.join(self.pdf_download_dir, f"{pmid}.pdf")
                    if os.path.exists(pdf_path):
                        article.pdf_path = pdf_path
                        candidate_pmids.append(pmid)
        
        source_desc = "FTS accepted" if self.config.get('fts_enabled') else "TIAB accepted"
        if not candidate_pmids:
            print(f"No articles found for Molecule Extraction from {source_desc} articles.")
            return []
            
        print(f"Found {len(candidate_pmids)} {source_desc} articles for Molecule Extraction.")
        return sorted(candidate_pmids)

    def _extract_molecules_from_article(self, article: PubMedArticle) -> Tuple[Optional[List[str]], Optional[str]]:
        """Extracts molecules from an article's PDF."""
        if not article.pdf_path or not os.path.exists(article.pdf_path):
            return None, f"Error - PDF file not found"
            
        full_text = extract_pdf_text(article.pdf_path)
        
        if full_text is None:
            return None, f"Error - PDF Text Extraction Failed ({os.path.basename(article.pdf_path)})"
        elif full_text == "":
            return None, f"Error - PDF Text Extraction Empty ({os.path.basename(article.pdf_path)})"
            
        print(f"  Sending extracted text (approx {len(full_text)} chars) to Ollama for Molecule Extraction...")
        try:
            # Format template with full_text
            specific_prompt = self.mol_prompt_template.format(full_text=full_text)
            full_prompt = f"Research Question: {self.research_question}\n\n---\n\n{specific_prompt}"
        except KeyError as e:
            return None, f"Error - Missing placeholder {{{e}}} in molecule extraction prompt template."
        except Exception as e:
            return None, f"Error - Failed to format molecule extraction prompt: {e}"

        response_text, error_message = self.ollama_client.generate(full_prompt)
        if error_message:
            return None, error_message
            
        if response_text:
            molecules = [line.strip() for line in response_text.splitlines() if line.strip() and line.strip().lower() != "none"]
            if molecules:
                print(f"  Extracted Molecules: {', '.join(molecules)}")
                return molecules, None
            else:
                print("  No specific molecules identified by LLM.")
                return [], None
        else:
            return None, "Error - Empty response received from Ollama during molecule extraction"

    def _update_article_with_molecules(self, article: PubMedArticle, molecules: List[str], error_message: Optional[str] = None) -> int:
        """Updates article with extracted molecules and writes to results file."""
        count = 0
        
        if error_message:
            return count
            
        # Update the article's molecules list
        article.molecules = molecules
        
        # Write to the results file
        if molecules:
            try:
                with open(self.output_file, 'a') as f:
                    for molecule in molecules:
                        line = f"{article.pmid}: {molecule}\n"
                        f.write(line)
                        count += 1
            except Exception as e:
                print(f"  Error writing to output file: {e}")
        
        # Save updated article data
        self.articles[article.pmid] = article
        save_articles_to_json(self.articles, self.articles_file)
        
        return count

    def run(self) -> None:
        """Executes the Molecule Extraction workflow."""
        print("\n--- Starting Molecule Extraction Phase ---")
        if not self.config.get('mol_extraction_enabled'):
            print("Molecule Extraction is disabled. Skipping.")
            return
            
        if not os.path.exists(self.pdf_download_dir):
            print(f"Error: PDF directory not found: {self.pdf_download_dir}. Cannot perform Molecule Extraction.")
            return
            
        pmids_to_process = self._get_articles_for_mol_extraction()
        if not pmids_to_process:
            print("No articles found for Molecule Extraction processing.")
            return
            
        total_to_process = len(pmids_to_process)
        run_processed_count = 0
        run_extraction_errors = 0
        total_molecules_extracted = 0
        
        for i, pmid in enumerate(pmids_to_process):
            article = self.articles[pmid]
            print(f"\n[{i + 1}/{total_to_process}] Processing Molecule Extraction for PMID: {pmid}")
            print(f"  Title: {article.title[:100]}{'...' if len(article.title) > 100 else ''}")
            
            time.sleep(self.delay)
            molecules, error_message = self._extract_molecules_from_article(article)
            
            if error_message:
                print(f"  Skipping PMID {pmid} due to error: {error_message}")
                run_extraction_errors += 1
            elif molecules is not None:
                written_count = self._update_article_with_molecules(article, molecules)
                total_molecules_extracted += written_count
            else:
                print(f"  Skipping PMID {pmid} due to unknown molecule extraction error.")
                run_extraction_errors += 1
                
            run_processed_count += 1
            
        print("\n--- Molecule Extraction Phase Complete ---")
        print(f"\n--- Molecule Extraction Run Summary ---")
        print(f"Total articles processed/attempted: {run_processed_count}")
        print(f"Total 'PMID: molecule' lines written: {total_molecules_extracted}")
        print(f"Articles skipped due to errors: {run_extraction_errors}")
        
        print(f"\nMolecule extraction results saved to '{self.output_file}'.")
        print("-----------------------------------------") 