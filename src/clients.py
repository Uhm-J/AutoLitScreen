import json
import requests
from Bio import Entrez
from typing import Dict, List, Optional, Tuple
from .utils import create_directory
from .models import PubMedArticle
import os

class PubMedClient:
    """Handles interactions with the NCBI PubMed Entrez API."""
    def __init__(self, email: str):
        if not email or email == "your.email@example.com": raise ValueError("Valid email needed for PubMedClient.")
        Entrez.email = email; print(f"PubMedClient initialized. Email: {email}")
    
    def fetch_details_batch(self, pmid_batch: List[str]) -> Dict[str, PubMedArticle]:
        """Fetches details for a batch of PMIDs and returns PubMedArticle objects."""
        if not pmid_batch: return {}
        print(f"Fetching details for batch of {len(pmid_batch)} PMIDs...")
        ids_str = ",".join(pmid_batch)
        articles: Dict[str, PubMedArticle] = {}
        try:
            handle = Entrez.efetch(db="pubmed", id=ids_str, rettype="xml", retmode="text")
            records = Entrez.read(handle)
            handle.close()
            pubmed_articles = records.get('PubmedArticle', [])
            if not isinstance(pubmed_articles, list): pubmed_articles = [pubmed_articles]
            for record in pubmed_articles:
                pmid: Optional[str] = None; title: Optional[str] = "No Title Found"; abstract: Optional[str] = "No Abstract Found"; doi: Optional[str] = None
                try:
                    medline_citation = record.get('MedlineCitation', {})
                    pmid = str(medline_citation.get('PMID', ''))
                    
                    article_info = medline_citation.get('Article', {})
                    title = article_info.get('ArticleTitle', 'No Title Found')
                    
                    abstract_section = article_info.get('Abstract', {})
                    abstract_text = abstract_section.get('AbstractText', None)
                    
                    if isinstance(abstract_text, list): abstract = "\n".join(str(part) for part in abstract_text if part)
                    elif isinstance(abstract_text, str): abstract = abstract_text
                    else: abstract = "No Abstract Found"
                    
                    # Extract DOI
                    for article_id in article_info.get('ELocationID', []):
                        try:
                            if hasattr(article_id, 'attributes') and article_id.attributes.get('EIdType') == 'doi':
                                doi = str(article_id).strip()
                                break
                        except Exception as e:
                            print(f"  Warning: Error parsing article ID: {e}")
                            continue
                    
                    # Extract journal info
                    journal = article_info.get('Journal', {})
                    journal_title = journal.get('Title', '')
                    
                    # Extract publication date
                    pub_date = None
                    journal_issue = journal.get('JournalIssue', {})
                    pub_date_elem = journal_issue.get('PubDate', {})
                    year = None
                    if 'Year' in pub_date_elem:
                        try:
                            year = int(pub_date_elem['Year'])
                        except (ValueError, TypeError):
                            year = 0
                    
                    # Extract volume and issue
                    volume = journal_issue.get('Volume', '')
                    issue = journal_issue.get('Issue', '')
                    
                    # Extract pagination
                    pagination = article_info.get('Pagination', {})
                    pages = pagination.get('MedlinePgn', '')
                    
                    # Extract authors
                    authors = []
                    author_list = article_info.get('AuthorList', [])
                    if author_list:
                        for author in author_list:
                            if 'LastName' in author and 'ForeName' in author:
                                author_name = f"{author['LastName']} {author['ForeName']}"
                                authors.append(author_name)
                            elif 'LastName' in author:
                                authors.append(author['LastName'])
                            elif 'CollectiveName' in author:
                                authors.append(author['CollectiveName'])
                    
                    # Create PubMedArticle object
                    if pmid:
                        articles[pmid] = PubMedArticle(
                            title=title,
                            authors=authors,
                            journal=journal_title,
                            year=year if year else 0,
                            volume=volume,
                            issue=issue,
                            pages=pages,
                            doi=doi if doi else "",
                            pmid=pmid,
                            abstract=abstract
                        )
                        
                except Exception as parse_e: 
                    print(f"  Warning: Error parsing details for PMID {pmid}: {parse_e}")
                    if pmid:
                        articles[pmid] = PubMedArticle(
                            title="Error Parsing Title",
                            authors=[],
                            journal="",
                            year=0,
                            abstract="Error Parsing Abstract",
                            pmid=pmid,
                            status_tiab="error"
                        )
            
            print(f"Successfully processed {len(articles)} articles in batch.")
            fetched_pmids = set(articles.keys()); missing_pmids = set(pmid_batch) - fetched_pmids
            if missing_pmids: 
                print(f"Warning: Could not retrieve records for PMIDs: {', '.join(missing_pmids)}")
                for mp in missing_pmids:
                    articles[mp] = PubMedArticle(
                        title="Error Fetching Title",
                        authors=[],
                        journal="",
                        year=0,
                        abstract="Error Fetching Abstract",
                        pmid=mp,
                        status_tiab="error"
                    )
            return articles
        except Exception as e: 
            print(f"Critical Error fetching PubMed details batch: {e}")
            return {pmid: PubMedArticle(
                title="Error Fetching Title",
                authors=[],
                journal="",
                year=0,
                abstract="Error Fetching Abstract",
                pmid=pmid,
                status_tiab="error"
            ) for pmid in pmid_batch}

class OllamaClient:
    """Handles interactions with a local Ollama API endpoint."""
    def __init__(self, model_name: str, api_url: str, request_timeout: int = 180, num_ctx: int = 16384):
        self.model_name = model_name; self.api_url = api_url; self.timeout = request_timeout; self.num_ctx = num_ctx
        self.headers = {'Content-Type': 'application/json'}; print(f"OllamaClient initialized: model '{model_name}', URL {api_url}")
    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Sends a prompt to Ollama and returns the response."""
        payload = {"model": self.model_name, "prompt": prompt, "stream": False, "options": {"num_ctx": self.num_ctx}}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            llm_response = response_data.get('response', '').strip()
            if not llm_response: print("  Warning: Received empty response from Ollama."); return None, "Error - Empty Ollama Response"
            return llm_response, None
        except requests.exceptions.Timeout: return None, f"Error - Ollama Timeout after {self.timeout}s"
        except requests.exceptions.RequestException as e: return None, f"Error - Ollama Request Failed: {e}"
        except json.JSONDecodeError: return None, f"Error - Invalid Ollama JSON Response: {response.text[:200]}..."
        except Exception as e: return None, f"Error - Unexpected Ollama Error: {e}"

class PdfDownloader:
    """Handles downloading PDFs, primarily using the Unpaywall API."""
    def __init__(self, download_dir: str, unpaywall_email: str):
        self.download_dir = download_dir; self.unpaywall_email = unpaywall_email
        self.session = requests.Session(); self.session.headers.update({'User-Agent': f'Python TIABScreener Script (mailto:{unpaywall_email})'})
        if not create_directory(self.download_dir): print(f"Warning: Could not create PDF download directory: {self.download_dir}.")
        print(f"PdfDownloader initialized. Saving PDFs to: {self.download_dir}")
    
    def download_pdf(self, article: PubMedArticle) -> bool:
        """Attempts to find and download a legal OA PDF using Unpaywall. Saves as PMID.pdf."""
        pmid = article.pmid
        doi = article.doi
        
        if not doi: 
            print(f"  PDF Download: Skipping PMID {pmid} - No DOI found.")
            article.status_pdf_fetch = "failed"
            return False
            
        print(f"  PDF Download: Querying Unpaywall for DOI: {doi}...")
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={self.unpaywall_email}"
        pdf_url: Optional[str] = None
        try:
            response = self.session.get(unpaywall_url, timeout=30); response.raise_for_status(); data = response.json()
            best_oa_location = data.get('best_oa_location')
            if best_oa_location and best_oa_location.get('url_for_pdf'): 
                pdf_url = best_oa_location['url_for_pdf']
                article.pdf_link = pdf_url
                print(f"  PDF Download: Found OA PDF URL via Unpaywall ({best_oa_location.get('host_type', 'unknown')}): {pdf_url}")
            else: 
                print(f"  PDF Download: No OA PDF link found via Unpaywall for DOI {doi}.")
                article.status_pdf_fetch = "failed"
                return False
        except requests.exceptions.RequestException as e: 
            print(f"  PDF Download: Error querying Unpaywall API for DOI {doi}: {e}")
            article.status_pdf_fetch = "error"
            return False
        except Exception as e: 
            print(f"  PDF Download: Unexpected error during Unpaywall query for DOI {doi}: {e}")
            article.status_pdf_fetch = "error"
            return False
            
        if pdf_url:
            try:
                print(f"  PDF Download: Attempting to download from {pdf_url}...")
                pdf_response = self.session.get(pdf_url, timeout=120, stream=True)
                pdf_response.raise_for_status()
                
                content_type = pdf_response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type: 
                    print(f"  PDF Download: Warning - URL content-type not PDF ('{content_type}'). Saving anyway.")
                
                filename = f"{pmid}.pdf"
                filepath = os.path.join(self.download_dir, filename)
                
                with open(filepath, 'wb') as f:
                    [f.write(chunk) for chunk in pdf_response.iter_content(chunk_size=8192)]
                
                # Set the PDF path on the article object
                article.pdf_path = filepath
                article.status_pdf_fetch = "accepted"
                
                print(f"  PDF Download: Successfully saved PDF to {filepath}")
                return True
            except requests.exceptions.RequestException as e: 
                print(f"  PDF Download: Error downloading PDF from {pdf_url}: {e}")
                article.status_pdf_fetch = "failed"
                return False
            except IOError as e: 
                print(f"  PDF Download: Error saving PDF file to {filepath}: {e}")
                article.status_pdf_fetch = "error"
                return False
            except Exception as e: 
                print(f"  PDF Download: Unexpected error during PDF download/save: {e}")
                article.status_pdf_fetch = "error"
                return False
        
        article.status_pdf_fetch = "failed"
        return False 