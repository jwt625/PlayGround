const state = {
  datasets: {},
  activeDataset: "3dj",
  rows: [],
  filtered: [],
  selected: null,
  selectedMonths: new Set(),
  activeTab: "overview",
  textIndexReady: false,
  textIndexLoading: false,
};

const els = {
  datasetSubtitle: document.querySelector("#datasetSubtitle"),
  datasetTabs: document.querySelectorAll(".dataset-tab"),
  stats: document.querySelector("#stats"),
  search: document.querySelector("#search"),
  sortBy: document.querySelector("#sortBy"),
  textSearch: document.querySelector("#textSearch"),
  monthList: document.querySelector("#monthList"),
  monthPanelTitle: document.querySelector("#monthPanelTitle"),
  clearMonths: document.querySelector("#clearMonths"),
  presentationList: document.querySelector("#presentationList"),
  resultCount: document.querySelector("#resultCount"),
  detailCode: document.querySelector("#detailCode"),
  detailTitle: document.querySelector("#detailTitle"),
  detailMeta: document.querySelector("#detailMeta"),
  detailLinks: document.querySelector("#detailLinks"),
  overviewTab: document.querySelector("#overviewTab"),
  markdownTab: document.querySelector("#markdownTab"),
  rawTab: document.querySelector("#rawTab"),
  overviewView: document.querySelector("#overviewView"),
  markdownView: document.querySelector("#markdownView"),
  rawView: document.querySelector("#rawView"),
};

const DATASETS = {
  "3dj": {
    label: "P802.3dj",
    subtitle: "Cached IEEE P802.3dj public meeting materials, metadata, and extracted text.",
    url: "metadata/talks.json",
    payloadKey: "talks",
    groupLabel: "Months",
    fileNoun: "PDFs",
    hasExtractedText: true,
  },
  e4ai: {
    label: "E4AI",
    subtitle: "Cached IEEE 802.3 Ethernet for AI Assessment public materials, metadata, and extracted text.",
    url: "../ieee802_e4ai_metadata/documents.json",
    payloadKey: "documents",
    groupLabel: "Pages",
    fileNoun: "files",
    hasExtractedText: true,
  },
};

const esc = (text = "") =>
  String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[ch]);

const pathFromApp = (path) => (path ? `../${path}` : "");
const compact = (value = "") => String(value).replace(/\s+/g, " ").trim();
const firstValue = (...values) => values.find((value) => compact(value)) || "";
const extensionLabel = (filename = "") => (filename.split(".").pop() || "file").toUpperCase();

function activeConfig() {
  return DATASETS[state.activeDataset];
}

function meetingSortValue(meeting) {
  if (meeting === "public_index") return 99991231;
  if (meeting === "channel") return 99991230;
  const workshopDate = /(\d{4})-(\d{2})-(\d{2})/.exec(meeting || "");
  if (workshopDate) return Number(`${workshopDate[1]}${workshopDate[2]}${workshopDate[3]}`);
  const match = /^(\d{2})_(\d{2})(\d{2})?$/.exec(meeting || "");
  if (!match) return 0;
  return Number(`20${match[1]}${match[2]}${match[3] || "00"}`);
}

function meetingLabel(meeting) {
  if (meeting === "public_index") return "Public index";
  if (meeting === "channel") return "Channel data";
  const workshopDate = /(\d{4})-(\d{2})-(\d{2})/.exec(meeting || "");
  if (workshopDate) {
    const date = new Date(Date.UTC(Number(workshopDate[1]), Number(workshopDate[2]) - 1, Number(workshopDate[3])));
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  }
  const match = /^(\d{2})_(\d{2})(\d{2})?$/.exec(meeting || "");
  if (!match) return meeting || "Unknown";
  const date = new Date(Date.UTC(2000 + Number(match[1]), Number(match[2]) - 1, Number(match[3] || "1")));
  const label = date.toLocaleString(undefined, { month: "short", year: "numeric", timeZone: "UTC" });
  return match[3] ? `${label} ${Number(match[3])}` : label;
}

function fileSize(bytes) {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}

function displayTitle(row) {
  return firstValue(row.website_title, row.title, row.filename);
}

function displayCode(row) {
  return firstValue(row.presentation_code, row.file_code, row.stem, row.filename);
}

function displayPeople(row) {
  return (row.presenters || []).join(", ");
}

function displayAffiliations(row) {
  return [...new Set(row.affiliations || [])].join(", ");
}

function baseSearchText(row) {
  return [
    row.meeting,
    meetingLabel(row.meeting),
    row.filename,
    row.stem,
    row.title,
    row.website_title,
    row.title_cell_text,
    row.presentation_code,
    row.file_code,
    row.presentation_date,
    row.presentation_date_iso,
    displayPeople(row),
    displayAffiliations(row),
    row.source_url,
    row.presentation_url,
    row.dataset,
    row.page_title,
    row.page_kind,
  ].join(" ").toLowerCase();
}

function normalize3dj(row) {
  return {
    ...row,
    dataset: "3dj",
    source_size: row.source_size || 0,
    source_url: row.source_url || row.presentation_url,
    source_path: row.source_path || "",
  };
}

function normalizeE4ai(row) {
  const filename = row.path ? row.path.split("/").pop() : (row.url || "").split("/").pop() || "";
  const stem = filename.replace(/\.[^.]+$/, "");
  return {
    ...row,
    dataset: "e4ai",
    meeting: row.page_slug,
    filename,
    stem,
    website_title: row.title,
    title_cell_text: row.title,
    presentation_code: "",
    file_code: "",
    presentation_url: row.url,
    source_url: row.url,
    source_path: row.path,
    source_size: row.size || 0,
    markdown_path: row.markdown_path || "",
    text_path: row.text_path || "",
    marker_output_dir: "",
  };
}

async function loadTextIndex() {
  if (state.textIndexReady || state.textIndexLoading) return;
  state.textIndexLoading = true;
  els.stats.textContent = "Indexing extracted text...";
  const chunkSize = 24;
  for (let index = 0; index < state.rows.length; index += chunkSize) {
    const chunk = state.rows.slice(index, index + chunkSize);
    await Promise.all(chunk.map(async (row) => {
      if (!row.text_path || row.fullText !== undefined) return;
      try {
        const response = await fetch(pathFromApp(row.text_path));
        row.fullText = response.ok ? (await response.text()).toLowerCase() : "";
      } catch {
        row.fullText = "";
      }
    }));
  }
  state.textIndexReady = true;
  state.textIndexLoading = false;
  updateStats();
  applyFilters();
}

function textExcerpt(row, query) {
  if (!query || !row.fullText) return "";
  const index = row.fullText.indexOf(query.toLowerCase());
  if (index < 0) return "";
  const start = Math.max(0, index - 90);
  const end = Math.min(row.fullText.length, index + query.length + 140);
  return compact(row.fullText.slice(start, end));
}

function sortRows(rows) {
  const mode = els.sortBy.value;
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
  return rows.sort((a, b) => {
    if (mode === "date-asc" || mode === "date-desc") {
      const av = a.presentation_date_iso || `${meetingSortValue(a.meeting)}`;
      const bv = b.presentation_date_iso || `${meetingSortValue(b.meeting)}`;
      return mode === "date-asc" ? collator.compare(av, bv) : collator.compare(bv, av);
    }
    if (mode === "meeting-asc" || mode === "meeting-desc") {
      const delta = meetingSortValue(a.meeting) - meetingSortValue(b.meeting);
      return mode === "meeting-asc" ? delta : -delta;
    }
    if (mode === "presenter-asc") {
      return collator.compare((a.presenters || [])[0] || "", (b.presenters || [])[0] || "");
    }
    if (mode === "size-desc") {
      return (b.source_size || 0) - (a.source_size || 0);
    }
    return collator.compare(displayTitle(a), displayTitle(b));
  });
}

function applyFilters() {
  const query = els.search.value.trim().toLowerCase();
  const includeText = els.textSearch.checked && state.textIndexReady;
  const monthLimited = state.selectedMonths.size > 0;

  state.filtered = sortRows(state.rows.filter((row) => {
    if (monthLimited && !state.selectedMonths.has(row.meeting)) return false;
    if (!query) return true;
    if (row.searchText.includes(query)) return true;
    return includeText && row.fullText && row.fullText.includes(query);
  }));

  renderMonths();
  renderList();
  if (!state.filtered.includes(state.selected)) {
    selectRow(state.filtered[0] || null);
  }
}

function monthCounts() {
  const counts = new Map();
  for (const row of state.rows) {
    counts.set(row.meeting, (counts.get(row.meeting) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => meetingSortValue(b[0]) - meetingSortValue(a[0]));
}

function renderMonths() {
  const query = els.search.value.trim().toLowerCase();
  els.monthList.innerHTML = monthCounts().map(([meeting, count]) => {
    const active = state.selectedMonths.has(meeting);
    const visible = state.filtered.filter((row) => row.meeting === meeting).length;
    return `
      <button class="month ${active ? "active" : ""}" type="button" data-meeting="${esc(meeting)}">
        <span class="month-check"></span>
        <span>
          <span class="month-name">${esc(meetingLabel(meeting))}</span>
          <span class="month-sub">${esc(meeting)}</span>
        </span>
        <span class="month-count">${query || state.selectedMonths.size ? esc(visible) + "/" : ""}${esc(count)}</span>
      </button>`;
  }).join("");

  els.monthList.querySelectorAll(".month").forEach((button) => {
    button.addEventListener("click", () => {
      const meeting = button.dataset.meeting;
      if (state.selectedMonths.has(meeting)) state.selectedMonths.delete(meeting);
      else state.selectedMonths.add(meeting);
      applyFilters();
    });
  });
}

function renderList() {
  els.resultCount.textContent = `${state.filtered.length}`;
  if (!state.filtered.length) {
    els.presentationList.innerHTML = `<p class="empty">No presentations match the current filters.</p>`;
    return;
  }
  const query = els.search.value.trim();
  els.presentationList.innerHTML = state.filtered.map((row, index) => {
    const excerpt = els.textSearch.checked ? textExcerpt(row, query) : "";
    return `
      <button class="presentation ${state.selected === row ? "active" : ""}" type="button" data-index="${index}">
        <span class="badge">${esc(displayCode(row))}</span>
        <span>
          <span class="presentation-title">${esc(displayTitle(row))}</span>
          <span class="presentation-sub">${esc(meetingLabel(row.meeting))} · ${esc(row.presentation_date || "No date")}</span>
          <span class="presentation-sub">${esc(displayPeople(row) || "No presenters listed")}</span>
          ${excerpt ? `<span class="presentation-sub">...${esc(excerpt)}...</span>` : ""}
          <span class="flags">
            <span class="flag">${esc(row.filename)}</span>
            ${row.markdown_path ? `<span class="flag">MD</span>` : `<span class="flag warn">MD</span>`}
            ${row.source_path ? `<span class="flag">${esc(extensionLabel(row.filename))}</span>` : `<span class="flag warn">FILE</span>`}
          </span>
        </span>
      </button>`;
  }).join("");

  els.presentationList.querySelectorAll(".presentation").forEach((button) => {
    button.addEventListener("click", () => selectRow(state.filtered[Number(button.dataset.index)]));
  });
}

function renderLinks(row) {
  if (!row) return "";
  const links = [];
  if (row.source_path) links.push(`<a href="${pathFromApp(row.source_path)}" target="_blank" rel="noreferrer">File</a>`);
  if (row.markdown_path) links.push(`<a href="${pathFromApp(row.markdown_path)}" target="_blank" rel="noreferrer">Markdown</a>`);
  if (row.text_path) links.push(`<a href="${pathFromApp(row.text_path)}" target="_blank" rel="noreferrer">Text</a>`);
  if (row.source_parent_url) links.push(`<a href="${esc(row.source_parent_url)}" target="_blank" rel="noreferrer">Meeting Page</a>`);
  if (row.presentation_url && row.presentation_url !== row.source_url) {
    links.push(`<a href="${esc(row.presentation_url)}" target="_blank" rel="noreferrer">Listed URL</a>`);
  }
  return links.join("");
}

function renderPeople(row) {
  const pairs = row.presenter_affiliations || [];
  if (!pairs.length) return `<p class="muted">No presenter metadata found.</p>`;
  return `
    <table class="people">
      <thead><tr><th>Presenter</th><th>Affiliation</th></tr></thead>
      <tbody>
        ${pairs.map((pair) => `<tr><td>${esc(pair.presenter)}</td><td>${esc(pair.affiliation)}</td></tr>`).join("")}
      </tbody>
    </table>`;
}

function renderOverview(row) {
  if (!row) return `<p class="empty">Select a presentation to see details.</p>`;
  return `
    <div class="overview-grid">
      <section class="info-block">
        <div class="label">Meeting</div>
        <div>${esc(meetingLabel(row.meeting))} <span class="muted">(${esc(row.meeting)})</span></div>
      </section>
      <section class="info-block">
        <div class="label">Date</div>
        <div>${esc(row.presentation_date || "Not listed")}</div>
      </section>
      <section class="info-block">
        <div class="label">Presentation Code</div>
        <div>${esc(displayCode(row) || "Not listed")}</div>
      </section>
      <section class="info-block">
        <div class="label">File</div>
        <div>${esc(row.filename)} <span class="muted">${esc(fileSize(row.source_size))}</span></div>
      </section>
      ${row.page_title ? `
      <section class="info-block full">
        <div class="label">Source Page</div>
        <div>${esc(row.page_title)} <span class="muted">${esc(row.page_kind || "")}</span></div>
      </section>` : ""}
      <section class="info-block full">
        <div class="label">Title Cell</div>
        <div>${esc(row.title_cell_text || displayTitle(row))}</div>
      </section>
      <section class="info-block full">
        <div class="label">People</div>
        ${renderPeople(row)}
      </section>
      <section class="info-block full">
        <div class="label">Sources</div>
        <p><span class="muted">Parent:</span> <a href="${esc(row.source_parent_url)}" target="_blank" rel="noreferrer">${esc(row.source_parent_url)}</a></p>
        <p><span class="muted">Presentation:</span> <a href="${esc(row.presentation_url || row.source_url)}" target="_blank" rel="noreferrer">${esc(row.presentation_url || row.source_url)}</a></p>
        <p><span class="muted">Cached file:</span> <a href="${pathFromApp(row.source_path)}" target="_blank" rel="noreferrer">${esc(row.source_path)}</a></p>
      </section>
    </div>`;
}

function renderRaw(row) {
  if (!row) return "";
  const fields = [
    "doc_id", "meeting", "filename", "stem", "title", "website_title", "title_cell_text",
    "presentation_code", "file_code", "presentation_date", "presentation_date_iso",
    "source_parent_url", "source_parent_path", "presentation_url", "source_url", "source_path",
    "source_size", "sha256", "markdown_path", "text_path", "dataset", "page_title", "page_kind", "status", "note",
  ];
  return `
    <table>
      <tbody>
        ${fields.map((field) => `<tr><th>${esc(field)}</th><td>${esc(row[field] ?? "")}</td></tr>`).join("")}
        <tr><th>presenters</th><td>${esc((row.presenters || []).join("\\n"))}</td></tr>
        <tr><th>affiliations</th><td>${esc((row.affiliations || []).join("\\n"))}</td></tr>
      </tbody>
    </table>`;
}

function inlineMarkdown(text) {
  let out = esc(text);
  out = out.replace(/&lt;(\/?)(sup|sub|em|strong)&gt;/g, "<$1$2>");
  out = out.replace(/&lt;br\s*\/?&gt;/g, "<br>");
  out = out.replace(/!\[[^\]]*\]\([^)]+\)/g, "");
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return out;
}

function isTableDivider(line) {
  return /^\s*\|?[\s:|-]+\|[\s:|.-]*$/.test(line);
}

function renderTable(lines) {
  const rows = lines.map((line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
  const [head, , ...body] = rows;
  return `
    <table>
      <thead><tr>${head.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead>
      <tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>`;
}

function renderMarkdown(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let list = [];
  let table = [];
  let code = [];
  let inCode = false;

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list.length) {
      html.push(`<ul>${list.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`);
      list = [];
    }
  };
  const flushTable = () => {
    if (table.length) {
      html.push(renderTable(table));
      table = [];
    }
  };

  for (const line of lines) {
    if (line.startsWith("```")) {
      if (inCode) {
        html.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
        code = [];
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        flushTable();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      code.push(line);
      continue;
    }
    if (line.trim().startsWith("|")) {
      flushParagraph();
      flushList();
      table.push(line);
      continue;
    }
    if (table.length && isTableDivider(line)) {
      table.push(line);
      continue;
    }
    flushTable();

    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length, 3);
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = /^\s*[-*]\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    paragraph.push(line.trim());
  }

  flushParagraph();
  flushList();
  flushTable();
  return html.join("\n");
}

async function selectRow(row) {
  state.selected = row;
  renderList();
  if (!row) {
    els.detailCode.textContent = "";
    els.detailTitle.textContent = "No presentation selected";
    els.detailMeta.textContent = "";
    els.detailLinks.innerHTML = "";
    els.overviewView.innerHTML = renderOverview(null);
    els.markdownView.innerHTML = "";
    els.rawView.innerHTML = "";
    return;
  }

  els.detailCode.textContent = displayCode(row);
  els.detailTitle.textContent = displayTitle(row);
  els.detailMeta.textContent = `${meetingLabel(row.meeting)} · ${row.presentation_date || "No date"} · ${displayPeople(row) || "No presenters listed"}`;
  els.detailLinks.innerHTML = renderLinks(row);
  els.overviewView.innerHTML = renderOverview(row);
  els.rawView.innerHTML = renderRaw(row);
  els.markdownView.innerHTML = row.markdown_path ? "<p>Loading extracted text...</p>" : "<p>No extracted markdown found.</p>";

  if (row.markdown_path) {
    try {
      const response = await fetch(pathFromApp(row.markdown_path));
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      els.markdownView.innerHTML = renderMarkdown(await response.text());
    } catch (error) {
      els.markdownView.innerHTML = `<p>Could not load markdown: ${esc(error.message)}</p>`;
    }
  }
}

function setTab(tab) {
  state.activeTab = tab;
  els.overviewTab.classList.toggle("active", tab === "overview");
  els.markdownTab.classList.toggle("active", tab === "markdown");
  els.rawTab.classList.toggle("active", tab === "raw");
  els.overviewView.classList.toggle("hidden", tab !== "overview");
  els.markdownView.classList.toggle("hidden", tab !== "markdown");
  els.rawView.classList.toggle("hidden", tab !== "raw");
}

function updateStats() {
  const meetings = new Set(state.rows.map((row) => row.meeting)).size;
  const mdCount = state.rows.filter((row) => row.markdown_path).length;
  const config = activeConfig();
  const textStatus = config.hasExtractedText ? (state.textIndexReady ? "text indexed" : "text ready") : "metadata only";
  els.stats.textContent = `${state.rows.length} ${config.fileNoun} · ${meetings} ${config.groupLabel.toLowerCase()} · ${mdCount} extracted · ${textStatus}`;
}

async function loadDataset(key) {
  if (state.datasets[key]) return state.datasets[key];
  const config = DATASETS[key];
  const response = await fetch(config.url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const data = await response.json();
  const normalizer = key === "e4ai" ? normalizeE4ai : normalize3dj;
  state.datasets[key] = (data[config.payloadKey] || []).map((row) => ({
    ...normalizer(row),
    searchText: baseSearchText(row),
  }));
  state.datasets[key].forEach((row) => {
    row.searchText = baseSearchText(row);
  });
  return state.datasets[key];
}

async function setDataset(key) {
  state.activeDataset = key;
  state.rows = await loadDataset(key);
  state.filtered = [];
  state.selected = null;
  state.selectedMonths.clear();
  state.textIndexReady = state.rows.every((row) => row.fullText !== undefined) && activeConfig().hasExtractedText;
  state.textIndexLoading = false;
  els.datasetSubtitle.textContent = activeConfig().subtitle;
  els.monthPanelTitle.textContent = activeConfig().groupLabel;
  els.textSearch.disabled = !activeConfig().hasExtractedText;
  els.textSearch.parentElement.classList.toggle("disabled", !activeConfig().hasExtractedText);
  if (!activeConfig().hasExtractedText) els.textSearch.checked = false;
  els.datasetTabs.forEach((button) => {
    button.classList.toggle("active", button.dataset.dataset === key);
  });

  updateStats();
  renderMonths();
  applyFilters();
  selectRow(state.filtered[0] || null);
}

async function init() {
  await setDataset("3dj");

  els.search.addEventListener("input", applyFilters);
  els.sortBy.addEventListener("change", applyFilters);
  els.textSearch.addEventListener("change", () => {
    if (!activeConfig().hasExtractedText) return;
    if (els.textSearch.checked) loadTextIndex();
    applyFilters();
  });
  els.clearMonths.addEventListener("click", () => {
    state.selectedMonths.clear();
    applyFilters();
  });
  els.overviewTab.addEventListener("click", () => setTab("overview"));
  els.markdownTab.addEventListener("click", () => setTab("markdown"));
  els.rawTab.addEventListener("click", () => setTab("raw"));
  els.datasetTabs.forEach((button) => {
    button.addEventListener("click", () => setDataset(button.dataset.dataset));
  });
}

init().catch((error) => {
  els.stats.textContent = "Error";
  els.presentationList.innerHTML = `<p class="empty">Could not load metadata/talks.json: ${esc(error.message)}</p>`;
});
