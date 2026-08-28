import { categories, equipment } from "./data.js";

const ASSET_PATH = "public/assets/equipment/";
const ISSUE_URL = "https://github.com/jwt625/PlayGround/issues/new";

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

function cardTemplate(item) {
  return `
    <article class="equipment-card">
      <div class="card-image">
        <img src="${ASSET_PATH}${escapeHtml(item.image)}" alt="Source-document photograph of ${escapeHtml(item.name)}" loading="lazy" />
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
          <a class="button icon-button" href="${commentUrl(item)}" target="_blank" rel="noreferrer" aria-label="Comment or ask about ${escapeHtml(item.name)}" title="Comment or ask">↗</a>
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
  resultCount.textContent = `${items.length} of ${equipment.length} systems`;
  emptyState.hidden = items.length !== 0;
  grid.hidden = items.length === 0;
}

function showDetails(item) {
  dialogContent.innerHTML = `
    <div class="dialog-layout">
      <div class="dialog-visual"><img src="${ASSET_PATH}${escapeHtml(item.image)}" alt="Source-document photograph of ${escapeHtml(item.name)}" /></div>
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
        </dl>
        <p class="condition"><strong>Audit note:</strong> ${escapeHtml(item.condition)}</p>
        <div class="dialog-actions">
          <a class="button button-primary" href="${commentUrl(item)}" target="_blank" rel="noreferrer">Comment / ask</a>
        </div>
        <p class="source-note">Audit reference: page ${item.page}. Comments open a prefilled GitHub issue. A GitHub account is required; no custom server is needed.</p>
      </div>
    </div>`;
  dialog.showModal();
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
  if (item) showDetails(item);
});

search.addEventListener("input", render);
document.querySelector("#clear-filters").addEventListener("click", () => {
  activeCategory = "All";
  search.value = "";
  renderFilters();
  render();
  search.focus();
});
document.querySelector("#dialog-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

renderFilters();
render();
