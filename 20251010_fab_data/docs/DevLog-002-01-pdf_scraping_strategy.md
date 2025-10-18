# PDF Scraping Strategy for Superconducting Qubits Research

## Overview
This document details the strategy for Phase 2 of the scraping plan: downloading and storing PDFs from lab websites with comprehensive metadata tracking.

**Goal**: Collect all available PDFs (theses, papers, presentations) from target labs with full metadata for later processing.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Source Inspection & Configuration (Phase 1)                 │
│ - Inspect each lab website                                   │
│ - Create YAML with CSS selectors, parsing rules              │
│ - Determine Playwright needs                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PDF Scraper (Playwright-based)                              │
│ - Navigate to lab pages                                      │
│ - Extract PDF links using CSS selectors                      │
│ - Download PDFs with retry logic                             │
│ - Calculate metadata (size, hash, timestamp)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Local Storage & Metadata Tracking                            │
│ - Store PDFs in organized directory structure                │
│ - Track in SQLite/JSON with full metadata                    │
│ - Log scraping status and errors                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Metadata Extraction (Future)                        │
│ - Extract text from PDFs (docling/marker)                    │
│ - Parse metadata (title, authors, year, advisor)             │
│ - Use AI agents for structured data extraction               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Source Inspection & Configuration (Phase 1)

### 2.1 Inspection Checklist
For each lab source, create a YAML configuration documenting:

```yaml
lab_id: yale_schoelkopf
inspection_status: COMPLETED  # PENDING, IN_PROGRESS, COMPLETED, BLOCKED
inspection_date: 2025-10-18
inspector_notes: "Static HTML, CSS selectors work well"

pages:
  publications:
    url: "https://rsl.yale.edu/publications"
    content_type: "static_html"  # static_html, dynamic_js, pdf_list, custom
    pdf_links_selector: "a.pdf-link"  # CSS selector for PDF links
    metadata_selectors:
      title: "h3.pub-title"
      authors: "span.authors"
      year: "span.year"
      doi: "span.doi"
    parsing_strategy: "css_selectors"  # css_selectors, regex, playwright_js
    requires_playwright: false
    requires_login: false
    robots_txt_compliant: true
    rate_limit: "1 request per 2 seconds"
    notes: "Straightforward structure"

  theses:
    url: "https://rsl.yale.edu/theses"
    content_type: "static_html"
    pdf_links_selector: "a[href$='.pdf']"
    metadata_selectors:
      title: "h4.thesis-title"
      author: "span.student-name"
      year: "span.graduation-year"
      advisor: "span.advisor-name"
    parsing_strategy: "css_selectors"
    requires_playwright: false
    requires_login: false
    robots_txt_compliant: true
    rate_limit: "1 request per 2 seconds"
    notes: "Advisor info in structured format"

  people:
    url: "https://rsl.yale.edu/people"
    content_type: "static_html"
    requires_playwright: false
    notes: "For Phase 4 (relationship mapping)"
```

### 2.2 Inspection Output
Create `sources_inspection.yaml` documenting:
- Content type (static HTML, dynamic JS, PDF list, custom)
- CSS selectors for PDF links and metadata
- Whether Playwright is needed
- Login requirements
- Rate limiting recommendations
- Any special parsing rules

---

## 3. PDF Scraper Implementation

### 3.1 Core Components

#### 3.1.1 Configuration Loader
```python
class SourceConfig:
    """Load and validate source configuration"""
    def __init__(self, yaml_path: str):
        self.config = self._load_yaml(yaml_path)
    
    def get_pdf_links_selector(self, lab_id: str, page: str) -> str:
        """Get CSS selector for PDF links"""
        return self.config[lab_id]['pages'][page]['pdf_links_selector']
    
    def requires_playwright(self, lab_id: str, page: str) -> bool:
        """Check if Playwright is needed"""
        return self.config[lab_id]['pages'][page]['requires_playwright']
```

#### 3.1.2 Playwright-based Scraper
```python
class PDFScraper:
    """Download PDFs from lab websites using Playwright"""
    
    def __init__(self, config: SourceConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.browser = None
        self.metadata_db = {}
    
    async def scrape_lab(self, lab_id: str):
        """Scrape all PDFs from a lab"""
        async with async_playwright() as p:
            self.browser = await p.firefox.launch()
            
            for page_name, page_config in self.config.get_pages(lab_id).items():
                await self.scrape_page(lab_id, page_name, page_config)
            
            await self.browser.close()
    
    async def scrape_page(self, lab_id: str, page_name: str, page_config: dict):
        """Scrape PDFs from a single page"""
        page = await self.browser.new_page()
        
        try:
            await page.goto(page_config['url'], wait_until='networkidle')
            
            # Extract PDF links
            pdf_links = await page.query_selector_all(
                page_config['pdf_links_selector']
            )
            
            for link in pdf_links:
                pdf_url = await link.get_attribute('href')
                await self.download_pdf(lab_id, page_name, pdf_url, page_config)
                
                # Random delay between downloads
                await asyncio.sleep(random.uniform(2, 5))
        
        finally:
            await page.close()
    
    async def download_pdf(self, lab_id: str, page_name: str, 
                          pdf_url: str, page_config: dict):
        """Download a single PDF with metadata"""
        try:
            # Resolve relative URLs
            full_url = urljoin(page_config['url'], pdf_url)
            
            # Download with timeout and retry
            response = await self._download_with_retry(full_url)
            
            if response.status_code == 200:
                # Generate local path
                local_path = self._generate_local_path(lab_id, page_name, full_url)
                
                # Save PDF
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                # Calculate metadata
                metadata = {
                    'work_id': self._generate_work_id(full_url),
                    'lab_id': lab_id,
                    'page_name': page_name,
                    'url_pdf': full_url,
                    'pdf_local_path': local_path,
                    'pdf_file_size': len(response.content),
                    'pdf_hash': hashlib.sha256(response.content).hexdigest(),
                    'scrape_timestamp': datetime.now().isoformat(),
                    'source_host': urlparse(full_url).netloc,
                    'source_url': page_config['url'],
                    'status': 'downloaded',
                    'access': 'open'
                }
                
                self.metadata_db[metadata['work_id']] = metadata
                self._log_success(metadata)
        
        except Exception as e:
            self._log_error(lab_id, page_name, pdf_url, str(e))
    
    async def _download_with_retry(self, url: str, max_retries: int = 3):
        """Download with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def _generate_local_path(self, lab_id: str, page_name: str, url: str) -> str:
        """Generate organized local storage path"""
        filename = os.path.basename(urlparse(url).path)
        if not filename.endswith('.pdf'):
            filename = f"{hashlib.md5(url.encode()).hexdigest()}.pdf"
        
        return os.path.join(
            self.output_dir,
            lab_id,
            page_name,
            filename
        )
    
    def _generate_work_id(self, url: str) -> str:
        """Generate unique work ID"""
        return hashlib.md5(url.encode()).hexdigest()
```

### 3.2 Metadata Tracking

#### 3.2.1 SQLite Schema
```sql
CREATE TABLE pdf_metadata (
    work_id TEXT PRIMARY KEY,
    lab_id TEXT NOT NULL,
    page_name TEXT NOT NULL,
    url_pdf TEXT NOT NULL,
    pdf_local_path TEXT NOT NULL,
    pdf_file_size INTEGER,
    pdf_hash TEXT UNIQUE,
    scrape_timestamp TEXT NOT NULL,
    source_host TEXT NOT NULL,
    source_url TEXT NOT NULL,
    status TEXT DEFAULT 'downloaded',  -- downloaded, failed, restricted
    access TEXT DEFAULT 'open',  -- open, restricted
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_lab_id ON pdf_metadata(lab_id);
CREATE INDEX idx_status ON pdf_metadata(status);
CREATE INDEX idx_pdf_hash ON pdf_metadata(pdf_hash);
```

#### 3.2.2 Metadata Storage
```python
class MetadataStore:
    """Store and query PDF metadata"""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def save_metadata(self, metadata: dict):
        """Save PDF metadata to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO pdf_metadata
            (work_id, lab_id, page_name, url_pdf, pdf_local_path,
             pdf_file_size, pdf_hash, scrape_timestamp, source_host,
             source_url, status, access)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata['work_id'],
            metadata['lab_id'],
            metadata['page_name'],
            metadata['url_pdf'],
            metadata['pdf_local_path'],
            metadata['pdf_file_size'],
            metadata['pdf_hash'],
            metadata['scrape_timestamp'],
            metadata['source_host'],
            metadata['source_url'],
            metadata['status'],
            metadata['access']
        ))
        self.conn.commit()
    
    def get_by_hash(self, pdf_hash: str):
        """Find PDF by hash (for deduplication)"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM pdf_metadata WHERE pdf_hash = ?", (pdf_hash,))
        return cursor.fetchone()
    
    def get_status_summary(self):
        """Get summary of scraping status"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM pdf_metadata
            GROUP BY status
        """)
        return dict(cursor.fetchall())
```

---

## 4. Execution Strategy

### 4.1 Overnight Scraping
```python
class OvernightScraper:
    """Run scraping overnight with random intervals"""
    
    def __init__(self, config_path: str, output_dir: str):
        self.config = SourceConfig(config_path)
        self.scraper = PDFScraper(self.config, output_dir)
        self.metadata_store = MetadataStore(f"{output_dir}/metadata.db")
    
    async def run_all_labs(self):
        """Scrape all labs with random intervals"""
        labs = self.config.get_all_labs()
        
        for lab_id in labs:
            try:
                print(f"Starting scrape for {lab_id}...")
                await self.scraper.scrape_lab(lab_id)
                
                # Save metadata
                for work_id, metadata in self.scraper.metadata_db.items():
                    self.metadata_store.save_metadata(metadata)
                
                # Random interval between labs (30 min to 2 hours)
                wait_time = random.uniform(1800, 7200)
                print(f"Waiting {wait_time/60:.1f} minutes before next lab...")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                print(f"Error scraping {lab_id}: {e}")
                continue
    
    def generate_report(self):
        """Generate scraping report"""
        summary = self.metadata_store.get_status_summary()
        print("\n=== Scraping Report ===")
        for status, count in summary.items():
            print(f"{status}: {count}")
```

### 4.2 Logging & Monitoring
```python
class ScrapingLogger:
    """Log scraping activities"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log_success(self, metadata: dict):
        """Log successful download"""
        with open(f"{self.log_dir}/success.log", "a") as f:
            f.write(f"{metadata['scrape_timestamp']} | {metadata['work_id']} | {metadata['url_pdf']}\n")
    
    def log_error(self, lab_id: str, page_name: str, url: str, error: str):
        """Log download error"""
        with open(f"{self.log_dir}/errors.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} | {lab_id}/{page_name} | {url} | {error}\n")
```

---

## 5. Quality Assurance

### 5.1 Validation Checks
- [ ] PDF file size > 0 bytes
- [ ] PDF hash is unique (no duplicates)
- [ ] Local path is accessible
- [ ] Source URL is valid
- [ ] Scrape timestamp is recent

### 5.2 Error Handling
- [ ] Network timeouts → retry with exponential backoff
- [ ] 404 errors → log and skip
- [ ] Access denied (403) → mark as restricted
- [ ] Corrupted PDFs → log and flag for manual review

---

## 6. Output Structure

```
output_dir/
├── yale_schoelkopf/
│   ├── publications/
│   │   ├── paper_001.pdf
│   │   ├── paper_002.pdf
│   │   └── ...
│   └── theses/
│       ├── thesis_001.pdf
│       ├── thesis_002.pdf
│       └── ...
├── yale_devoret/
│   ├── publications/
│   └── theses/
├── ...
├── metadata.db
├── logs/
│   ├── success.log
│   ├── errors.log
│   └── scraping_report.txt
└── sources_inspection.yaml
```

---

## 7. Next Steps (Phase 3)

After PDF scraping is complete:
1. Extract text from PDFs using docling or marker
2. Parse metadata (title, authors, year, advisor)
3. Use AI agents to extract structured data from CV/bio
4. Build authorship and advisor relationships

---

© 2025 Superconducting Qubit Knowledge Map Initiative

