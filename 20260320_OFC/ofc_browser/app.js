const state = {
  rows: [],
  filtered: [],
  selected: null,
  activeTab: "markdown",
};

const els = {
  stats: document.querySelector("#stats"),
  search: document.querySelector("#search"),
  sectionFilter: document.querySelector("#sectionFilter"),
  availableOnly: document.querySelector("#availableOnly"),
  paperList: document.querySelector("#paperList"),
  resultCount: document.querySelector("#resultCount"),
  detailCode: document.querySelector("#detailCode"),
  detailTitle: document.querySelector("#detailTitle"),
  detailMeta: document.querySelector("#detailMeta"),
  detailLinks: document.querySelector("#detailLinks"),
  detailNote: document.querySelector("#detailNote"),
  markdownTab: document.querySelector("#markdownTab"),
  tableTab: document.querySelector("#tableTab"),
  markdownView: document.querySelector("#markdownView"),
  rowView: document.querySelector("#rowView"),
};

const esc = (text = "") =>
  String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[ch]);

const pathFromApp = (path) => path ? `../${path}` : "";

function inlineMarkdown(text) {
  let out = esc(text);
  out = out.replace(/&lt;(\/?)(sup|sub|em|strong)&gt;/g, "<$1$2>");
  out = out.replace(/&lt;br\s*\/?&gt;/g, "<br>");
  out = out.replace(/&lt;span id=&quot;([^&]+)&quot;&gt;&lt;\/span&gt;/g, '<span id="$1"></span>');
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
  const rows = lines.map((line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const [head, , ...body] = rows;
  return `
    <table>
      <thead><tr>${head.map((c) => `<th>${inlineMarkdown(c)}</th>`).join("")}</tr></thead>
      <tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inlineMarkdown(c)}</td>`).join("")}</tr>`).join("")}</tbody>
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

function rowSearchText(row) {
  return [row.code, row.title, row.affiliation, row.section, row.note, row.abstract]
    .join(" ")
    .toLowerCase();
}

function applyFilters() {
  const query = els.search.value.trim().toLowerCase();
  const section = els.sectionFilter.value;
  const availableOnly = els.availableOnly.checked;
  state.filtered = state.rows.filter((row) => {
    if (section && row.section !== section) return false;
    if (availableOnly && !(row.hasPdf || row.hasMarkdown || row.hasText)) return false;
    if (query && !rowSearchText(row).includes(query)) return false;
    return true;
  });
  renderList();
}

function renderList() {
  els.resultCount.textContent = `${state.filtered.length}`;
  els.paperList.innerHTML = state.filtered.map((row, index) => `
    <button class="paper ${state.selected === row ? "active" : ""}" type="button" data-index="${index}">
      <span class="badge">${esc(row.code)}</span>
      <span>
        <div class="paper-title">${esc(row.title)}</div>
        <div class="paper-sub">${esc(row.section)}</div>
        <div class="paper-sub">${esc(row.affiliation)}</div>
        <div class="flags">
          <span class="flag ${row.hasPdf ? "" : "missing"}">PDF</span>
          <span class="flag ${row.hasMarkdown ? "" : "missing"}">MD</span>
          <span class="flag ${row.hasText ? "" : "missing"}">TXT</span>
        </div>
      </span>
    </button>
  `).join("");

  els.paperList.querySelectorAll(".paper").forEach((button) => {
    button.addEventListener("click", () => selectRow(state.filtered[Number(button.dataset.index)]));
  });
}

function renderLinks(row) {
  const links = [];
  if (row.pdfPath) links.push(`<a href="${pathFromApp(row.pdfPath)}" target="_blank" rel="noreferrer">PDF</a>`);
  if (row.markdownPath) links.push(`<a href="${pathFromApp(row.markdownPath)}" target="_blank" rel="noreferrer">Markdown</a>`);
  if (row.textPath) links.push(`<a href="${pathFromApp(row.textPath)}" target="_blank" rel="noreferrer">Text</a>`);
  return links.length ? links.join("") : `<span class="flag missing">No local files found</span>`;
}

function renderRowView(row) {
  return `
    <table>
      <tbody>
        <tr><th>Code</th><td>${esc(row.code)}</td></tr>
        <tr><th>Section</th><td>${esc(row.section)}</td></tr>
        <tr><th>Presentation / paper</th><td>${esc(row.title)}</td></tr>
        <tr><th>Affiliation(s)</th><td>${esc(row.affiliation)}</td></tr>
        <tr><th>${esc(row.noteLabel || "Note")}</th><td>${esc(row.note)}</td></tr>
        <tr><th>Abstract</th><td>${esc(row.abstract || row.description || "") || "Missing"}</td></tr>
        <tr><th>PDF</th><td>${row.pdfPath ? `<a href="${pathFromApp(row.pdfPath)}">${esc(row.pdfPath)}</a>` : "Missing"}</td></tr>
        <tr><th>Markdown</th><td>${row.markdownPath ? `<a href="${pathFromApp(row.markdownPath)}">${esc(row.markdownPath)}</a>` : "Missing"}</td></tr>
        <tr><th>Text</th><td>${row.textPath ? `<a href="${pathFromApp(row.textPath)}">${esc(row.textPath)}</a>` : "Missing"}</td></tr>
      </tbody>
    </table>`;
}

async function selectRow(row) {
  state.selected = row;
  renderList();
  els.detailCode.textContent = row.code;
  els.detailTitle.textContent = row.title;
  els.detailMeta.textContent = `${row.section} · ${row.affiliation || "No affiliation listed"}`;
  els.detailLinks.innerHTML = renderLinks(row);
  els.detailNote.innerHTML = `<strong>${esc(row.noteLabel || "Note")}:</strong> ${esc(row.note)}`;
  els.rowView.innerHTML = renderRowView(row);
  els.markdownView.innerHTML = row.markdownPath
    ? "<p>Loading extracted markdown...</p>"
    : renderFallback(row);

  if (row.markdownPath) {
    try {
      const response = await fetch(pathFromApp(row.markdownPath));
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const md = await response.text();
      els.markdownView.innerHTML = renderMarkdown(md);
    } catch (error) {
      els.markdownView.innerHTML = `<p>Could not load markdown: ${esc(error.message)}</p>`;
    }
  }
}

function renderFallback(row) {
  const abstract = row.abstract || row.description || "";
  if (!abstract) {
    return "<p>No extracted markdown or metadata abstract was found for this paper.</p>";
  }
  return `
    <h2>Abstract</h2>
    <p>${inlineMarkdown(abstract)}</p>
    <p><small>Shown from cached OFC metadata because no extracted markdown file was found.</small></p>`;
}

function setTab(tab) {
  state.activeTab = tab;
  els.markdownTab.classList.toggle("active", tab === "markdown");
  els.tableTab.classList.toggle("active", tab === "table");
  els.markdownView.classList.toggle("hidden", tab !== "markdown");
  els.rowView.classList.toggle("hidden", tab !== "table");
}

async function init() {
  const response = await fetch("papers.json");
  const data = await response.json();
  state.rows = data.rows;
  state.filtered = data.rows;

  const sections = [...new Set(state.rows.map((row) => row.section))].filter(Boolean);
  els.sectionFilter.insertAdjacentHTML(
    "beforeend",
    sections.map((section) => `<option value="${esc(section)}">${esc(section)}</option>`).join("")
  );
  const pdfCount = state.rows.filter((row) => row.hasPdf).length;
  const mdCount = state.rows.filter((row) => row.hasMarkdown).length;
  els.stats.textContent = `${state.rows.length} rows · ${pdfCount} PDFs · ${mdCount} markdown files`;

  els.search.addEventListener("input", applyFilters);
  els.sectionFilter.addEventListener("change", applyFilters);
  els.availableOnly.addEventListener("change", applyFilters);
  els.markdownTab.addEventListener("click", () => setTab("markdown"));
  els.tableTab.addEventListener("click", () => setTab("table"));

  applyFilters();
  if (state.rows.length) selectRow(state.rows[0]);
}

init().catch((error) => {
  els.stats.textContent = "Error";
  els.paperList.innerHTML = `<p class="paper-sub">Could not load papers.json: ${esc(error.message)}</p>`;
});
