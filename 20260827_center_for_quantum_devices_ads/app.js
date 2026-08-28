import { categories, equipment } from "./data.js";

const ASSET_PATH = "public/assets/equipment/";
const ISSUE_URL = "https://github.com/jwt625/cqd-equipment/issues/new";

const grid = document.querySelector("#equipment-grid");
const filters = document.querySelector("#category-filters");
const search = document.querySelector("#search");
const resultCount = document.querySelector("#result-count");
const emptyState = document.querySelector("#empty-state");
const dialog = document.querySelector("#item-dialog");
const dialogContent = document.querySelector("#dialog-content");
let activeCategory = "All";

document.querySelector("#item-count").textContent = equipment.length;

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);
}

function searchableText(item) {
  return Object.values(item).join(" ").toLocaleLowerCase();
}

function visibleItems() {
  const query = search.value.trim().toLocaleLowerCase();
  return equipment.filter((item) =>
    (activeCategory === "All" || item.category === activeCategory) &&
    (!query || searchableText(item).includes(query))
  );
}

function imageTemplate(item, detail = false) {
  if (!item.image) {
    return `<div class="image-placeholder${detail ? " image-placeholder-detail" : ""}"><span>No unambiguous audit photo</span></div>`;
  }
  return `<img src="${ASSET_PATH}${escapeHtml(item.image)}" alt="Audit photograph associated with ${escapeHtml(item.name)}"${detail ? "" : ' loading="lazy"'} />`;
}

function cardTemplate(item) {
  return `
    <article class="equipment-card">
      <div class="card-image">
        ${imageTemplate(item)}
        <span class="category-label">${escapeHtml(item.category)}</span>
      </div>
      <div class="card-body">
        <h3>${escapeHtml(item.name)}</h3>
        <p class="card-summary">${escapeHtml(item.summary)}</p>
        <dl class="card-meta">
          <div><dt>Maker</dt><dd title="${escapeHtml(item.manufacturer)}">${escapeHtml(item.manufacturer)}</dd></div>
          <div><dt>Model</dt><dd title="${escapeHtml(item.model)}">${escapeHtml(item.model)}</dd></div>
          <div><dt>Location</dt><dd title="${escapeHtml(item.location)}">${escapeHtml(item.location)}</dd></div>
          <div><dt>NU tag</dt><dd title="${escapeHtml(item.tag)}">${escapeHtml(item.tag)}</dd></div>
        </dl>
        <div class="card-actions">
          <button class="button" type="button" data-details="${item.id}">View details</button>
          <a class="button inquiry-button" href="${commentUrl(item)}" target="_blank" rel="noreferrer" aria-label="Ask a public question about ${escapeHtml(item.name)}">Public question ↗</a>
        </div>
      </div>
    </article>`;
}

function commentUrl(item) {
  const title = `[CQD equipment] ${item.name}`;
  const body = `Equipment: ${item.name}\nNU tag: ${item.tag}\nSource page: ${item.page}\n\nQuestion / comment:\n`;
  return `${ISSUE_URL}?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}&labels=CQD-equipment`;
}

function renderFilters() {
  filters.innerHTML = categories.map((category) => `
    <button class="filter-button" type="button" data-category="${escapeHtml(category)}" aria-pressed="${category === activeCategory}">
      ${escapeHtml(category)}
    </button>`).join("");
}

function render() {
  const items = visibleItems();
  grid.innerHTML = items.map(cardTemplate).join("");
  resultCount.textContent = `${items.length} of ${equipment.length} listings`;
  emptyState.hidden = items.length !== 0;
  grid.hidden = items.length === 0;
}

function showDetails(item) {
  dialogContent.innerHTML = `
    <div class="dialog-layout">
      <div class="dialog-visual">${imageTemplate(item, true)}</div>
      <div class="dialog-copy">
        <p class="eyebrow">${escapeHtml(item.category)} · Source page ${item.page}</p>
        <h2 id="dialog-title">${escapeHtml(item.name)}</h2>
        <p>${escapeHtml(item.summary)}</p>
        <dl class="facts">
          <div><dt>Manufacturer</dt><dd>${escapeHtml(item.manufacturer)}</dd></div>
          <div><dt>Model</dt><dd>${escapeHtml(item.model)}</dd></div>
          <div><dt>NU tag</dt><dd>${escapeHtml(item.tag)}</dd></div>
          <div><dt>Location</dt><dd>${escapeHtml(item.location)}</dd></div>
          <div><dt>Purchase date</dt><dd>${escapeHtml(item.purchaseDate)}</dd></div>
          ${item.serial ? `<div><dt>Serial number</dt><dd>${escapeHtml(item.serial)}</dd></div>` : ""}
          ${item.specification ? `<div><dt>Specification</dt><dd>${escapeHtml(item.specification)}</dd></div>` : ""}
        </dl>
        <p class="condition"><strong>${escapeHtml(item.noteType || "Audit note")}:</strong> ${escapeHtml(item.condition)}</p>
        <div class="dialog-actions">
          <a class="button button-primary" href="${commentUrl(item)}" target="_blank" rel="noreferrer">Ask a public question on GitHub</a>
        </div>
        <p class="source-note">Audit reference: page ${item.page}. Comments open a prefilled GitHub issue. A GitHub account is required; no custom server is needed.</p>
      </div>
    </div>`;
  if (!dialog.open) dialog.showModal();
}

function itemFromHash() {
  if (!window.location.hash.startsWith("#item=")) return null;
  const id = decodeURIComponent(window.location.hash.slice(6));
  return equipment.find((item) => item.id === id) || null;
}

function syncDialogToUrl() {
  const item = itemFromHash();
  if (item) showDetails(item);
  else if (dialog.open) dialog.close();
}

function closeDetails() {
  if (!dialog.open) return;
  dialog.close();
  if (window.location.hash.startsWith("#item=")) {
    history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  }
}

filters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-category]");
  if (!button) return;
  activeCategory = button.dataset.category;
  renderFilters();
  render();
});

grid.addEventListener("click", (event) => {
  const button = event.target.closest("[data-details]");
  if (!button) return;
  const item = equipment.find((candidate) => candidate.id === button.dataset.details);
  if (item) {
    history.pushState({ item: item.id }, "", `#item=${encodeURIComponent(item.id)}`);
    showDetails(item);
  }
});

search.addEventListener("input", render);
document.querySelector("#clear-filters").addEventListener("click", () => {
  activeCategory = "All";
  search.value = "";
  renderFilters();
  render();
  search.focus();
});
document.querySelector("#dialog-close").addEventListener("click", closeDetails);
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) closeDetails();
});
window.addEventListener("popstate", syncDialogToUrl);
window.addEventListener("hashchange", syncDialogToUrl);

renderFilters();
render();
syncDialogToUrl();
