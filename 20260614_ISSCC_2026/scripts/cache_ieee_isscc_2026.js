const fs = require("fs");
const path = require("path");
const { firefox } = require("playwright");

const PROCEEDINGS_URL =
  "https://ieeexplore.ieee.org/xpl/conhome/11408863/proceeding?sortType=vol-only-seq&isnumber=11408946&pageNumber=1";
const TOC_API =
  "https://ieeexplore.ieee.org/rest/search/pub/11408863/issue/11408946/toc";
const BASE = "https://ieeexplore.ieee.org";
const PUNUMBER = "11408863";
const ISNUMBER = "11408946";
const SORT_TYPE = "vol-only-seq";
const DEFAULT_FIREFOX =
  "/Users/wentaojiang/Library/Caches/ms-playwright/firefox-1509/firefox/Nightly.app/Contents/MacOS/firefox";

function cleanText(value) {
  return String(value ?? "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function absoluteUrl(link) {
  if (!link) return null;
  if (/^https?:\/\//i.test(link)) return link;
  return `${BASE}${link.startsWith("/") ? "" : "/"}${link}`;
}

function csvEscape(value) {
  const s = value == null ? "" : String(value);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function normalizeAuthor(author) {
  const id = author.id == null ? null : String(author.id);
  return {
    name: author.preferredName || author.searchablePreferredName || cleanText(`${author.firstName || ""} ${author.lastName || ""}`),
    normalized_name: author.normalizedName || null,
    first_name: author.firstName || null,
    last_name: author.lastName || null,
    ieee_author_id: id,
    url: id ? `${BASE}/author/${id}` : null,
  };
}

function normalizeRecord(record, pageNumber, indexOnPage) {
  const articleNumber = String(record.articleNumber || "");
  const doi = record.doi || (articleNumber ? `10.1109/ISSCC49663.2026.${articleNumber}` : null);
  const documentUrl = absoluteUrl(record.documentLink) || (articleNumber ? `${BASE}/document/${articleNumber}/` : null);
  const pdfUrl = absoluteUrl(record.pdfLink);
  const authors = (record.authors || []).map(normalizeAuthor);
  const title = cleanText(record.articleTitle);
  const sequenceMatch = title.match(/^(\d+(?:\.\d+)?)/);

  return {
    article_number: articleNumber || null,
    title,
    sequence: sequenceMatch ? sequenceMatch[1] : null,
    url: documentUrl,
    pdf_url: pdfUrl,
    doi,
    doi_url: doi ? `https://doi.org/${doi}` : null,
    publication_title: record.publicationTitle || record.displayPublicationTitle || null,
    publication_date: record.publicationDate || null,
    publication_year: record.publicationYear || null,
    volume: record.volume || null,
    start_page: record.startPage || null,
    end_page: record.endPage || null,
    content_type: record.articleContentType || record.contentType || null,
    access_type: record.accessType?.type || null,
    download_count: record.downloadCount ?? null,
    citation_count: record.citationCount ?? null,
    abstract: cleanText(record.abstract),
    authors,
    source_page_number: pageNumber,
    source_index_on_page: indexOnPage,
    raw_document_link: record.documentLink || null,
  };
}

async function postToc(page, pageNumber) {
  return await page.evaluate(
    async ({ url, body }) => {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json, text/plain, */*",
        },
        body: JSON.stringify(body),
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error(`TOC API ${response.status}: ${await response.text()}`);
      }
      return await response.json();
    },
    {
      url: TOC_API,
      body: {
        sortType: SORT_TYPE,
        isnumber: ISNUMBER,
        pageNumber: String(pageNumber),
        punumber: PUNUMBER,
      },
    }
  );
}

async function main() {
  const outDir = path.resolve("cache");
  fs.mkdirSync(outDir, { recursive: true });

  const executablePath = fs.existsSync(DEFAULT_FIREFOX) ? DEFAULT_FIREFOX : undefined;
  const browser = await firefox.launch({ headless: true, executablePath });
  const page = await browser.newPage({
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0",
  });

  try {
    await page.goto(PROCEEDINGS_URL, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForTimeout(4000);

    const first = await postToc(page, 1);
    const totalRecords = Number(first.totalRecords || first.total || first.recordCount || 0);
    const pageSize = first.records?.length || 25;
    const totalPages = totalRecords ? Math.ceil(totalRecords / pageSize) : 1;

    const pages = [{ pageNumber: 1, data: first }];
    for (let pageNumber = 2; pageNumber <= totalPages; pageNumber += 1) {
      pages.push({ pageNumber, data: await postToc(page, pageNumber) });
      await page.waitForTimeout(250);
    }

    const records = pages.flatMap(({ pageNumber, data }) =>
      (data.records || []).map((record, index) => normalizeRecord(record, pageNumber, index + 1))
    );

    const seen = new Set();
    const papers = records.filter((record) => {
      const key = record.article_number || record.doi || record.url || `${record.title}:${record.start_page}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    const cache = {
      cached_at: new Date().toISOString(),
      source: {
        proceedings_url: PROCEEDINGS_URL,
        toc_api: TOC_API,
        punumber: PUNUMBER,
        isnumber: ISNUMBER,
        sort_type: SORT_TYPE,
        reported_total_records: totalRecords || null,
        fetched_pages: pages.length,
        page_size: pageSize,
      },
      notes: [
        "Author URLs are IEEE Xplore author profile URLs derived from author IDs returned by the IEEE TOC API.",
        "Paper URLs are canonical IEEE Xplore document URLs derived from documentLink/articleNumber returned by the IEEE TOC API.",
      ],
      papers,
    };

    const jsonPath = path.join(outDir, "isscc_2026_ieee_xplore_papers.json");
    fs.writeFileSync(jsonPath, `${JSON.stringify(cache, null, 2)}\n`);

    const csvRows = [
      [
        "article_number",
        "sequence",
        "title",
        "url",
        "doi",
        "start_page",
        "end_page",
        "authors",
        "author_urls",
      ],
      ...papers.map((paper) => [
        paper.article_number,
        paper.sequence,
        paper.title,
        paper.url,
        paper.doi,
        paper.start_page,
        paper.end_page,
        paper.authors.map((author) => author.name).join("; "),
        paper.authors.map((author) => author.url).filter(Boolean).join("; "),
      ]),
    ];
    const csvPath = path.join(outDir, "isscc_2026_ieee_xplore_papers.csv");
    fs.writeFileSync(csvPath, `${csvRows.map((row) => row.map(csvEscape).join(",")).join("\n")}\n`);

    console.log(`Wrote ${papers.length} records`);
    console.log(jsonPath);
    console.log(csvPath);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
