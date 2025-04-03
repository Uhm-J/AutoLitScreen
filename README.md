# AutoLitScreen

An automated literature screening system powered by local LLMs (via Ollama) that processes scientific articles through a configurable, multi-phase workflow:

1. **TIAB Screening** - Evaluate articles based on titles and abstracts 
2. **PDF Fetching** - Download PDFs for accepted articles (via Unpaywall)
3. **Full-Text Screening** - Evaluate downloaded articles based on full text
4. **Molecule Extraction** - Extract molecule names from accepted articles

## Features

- **Configurable screening criteria** - Define your research question, inclusion and exclusion criteria
- **Multi-phase workflow** - Process articles through titles/abstracts first, then full-text screening
- **Local LLM-powered** - Uses Ollama to run inference on your local machine (privacy, no API costs)
- **PDF retrieval** - Automatically fetches PDFs for accepted articles via Unpaywall
- **Molecule extraction** - Identifies and extracts molecule names from relevant articles
- **JSON storage** - All article data stored in JSON format for easy processing
- **RIS export** - Export accepted articles to RIS format for reference managers

## Installation

### Prerequisites

- Python 3.x
- [Ollama](https://ollama.com/) installed and running
- An Ollama model pulled (e.g., `ollama pull llama3`)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/AutoLitScreen.git
   cd AutoLitScreen
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure your settings in `config.toml` (see Configuration section)

## Configuration

Edit the `config.toml` file to customize your screening process. Key settings to configure:

- **Research Question**: Define your specific research question
- **Email**: Required for PubMed API and Unpaywall (if using PDF fetching)
- **Ollama Settings**: Set the model name and URL for your local Ollama instance
- **Input/Output Paths**: Define locations for input PMIDs and output files
- **Screening Options**: Enable/disable different phases (TIAB, FTS, molecule extraction)
- **Criteria Files**: Point to your inclusion/exclusion criteria text files

A sample config.toml file is provided in the repository which you can customize for your needs.

## Creating Input Files

1. **PMID Input File**: Create a text file with one PubMed ID per line.

2. **Prompt Templates**: Create prompt template files for TIAB, FTS, and molecule extraction.
   - Include placeholders like `{inclusion}`, `{exclusion}`, `{title}`, `{abstract}`, and `{full_text}` as needed.

3. **Criteria Files**: Create separate files for inclusion and exclusion criteria.

## Usage

### Basic Screening

Run the complete screening workflow:

```
python main.py
```

This will:
1. Screen titles/abstracts from PMIDs in your input file
2. Download PDFs for accepted articles (if enabled)
3. Screen full texts of downloaded PDFs (if enabled)
4. Extract molecules from accepted articles (if enabled)

### Exporting Results

Export accepted articles to RIS format for import into reference managers and to retrieve full texts:

```
python main.py --ris
```

Exporting options:
- `--tiab-only`: Export only articles accepted in TIAB phase
- `--fts-only`: Export only articles accepted in FTS phase
- `-o FILENAME.ris`: Specify custom output filename

## Workflow Details

### TIAB Screening

1. PMIDs are loaded from the input file
2. Article metadata (title, abstract) is fetched from PubMed
3. For each article, the title and abstract are evaluated against criteria
4. Results are stored in the articles.json file

### Full-Text Screening

1. PDFs for TIAB-accepted articles are processed
2. Full text is extracted and evaluated against criteria
3. Results are stored in the articles.json file

### Molecule Extraction

1. Articles that passed FTS (or TIAB if FTS disabled) are processed
2. LLM identifies molecule names mentioned in the article
3. Results are stored in the articles.json file and exported to CSV

## Output Files

- `articles.json`: Complete database of all processed articles
- `accepted_articles.ris`: Exported citations in RIS format (when using --ris)
- `extracted_molecules.csv`: List of molecules extracted from accepted articles

