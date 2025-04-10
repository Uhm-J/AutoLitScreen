from dataclasses import dataclass, field
from typing import List, Optional, Literal

StatusType = Literal["accepted", "rejected", "failed", "error", "pending"]

@dataclass
class PubMedArticle:
    title: str
    authors: List[str]
    journal: str
    year: int
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    pmid: str = ""
    abstract: str = ""
    status_tiab: StatusType = "pending"
    tiab_notes: str = ""
    status_fts: StatusType = "pending"
    fts_notes: str = ""
    status_pdf_fetch: StatusType = "pending"
    pdf_link: Optional[str] = None
    molecules: List[str] = field(default_factory=list)
    pdf_path: Optional[str] = None

    def to_ris(self) -> str:
        ris = [
            "TY  - JOUR",
            *(f"AU  - {author}" for author in self.authors),
            f"TI  - {self.title}",
            f"JO  - {self.journal}",
            f"PY  - {self.year}",
            f"VL  - {self.volume}" if self.volume else "",
            f"IS  - {self.issue}" if self.issue else "",
            f"SP  - {self.pages.split('-')[0]}" if self.pages else "",
            f"EP  - {self.pages.split('-')[1]}" if '-' in self.pages else "",
            f"DO  - {self.doi}" if self.doi else "",
            f"AB  - {self.abstract}" if self.abstract else "",
            f"ID  - {self.pmid}",
            f"N1  - TIAB: {self.tiab_notes}" if self.tiab_notes else "",
            f"N2  - FTS: {self.fts_notes}" if self.fts_notes else "",
            f"UR  - https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/",
            "ER  -"
        ]
        return "\n".join(filter(None, ris))

    def get_molecules_formatted(self) -> str:
        """Returns a comma-separated list of extracted molecules"""
        return ", ".join(self.molecules) if self.molecules else "" 