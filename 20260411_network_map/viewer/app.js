const state = {
  datasets: [],
  activeDataset: null,
  records: [],
  filter: "all",
  filteredRecords: [],
  coverflowIndex: 0,
  coverflowSignature: "",
};

const elements = {
  datasetSelect: document.querySelector("#datasetSelect"),
  searchInput: document.querySelector("#searchInput"),
  sortSelect: document.querySelector("#sortSelect"),
  viewSelect: document.querySelector("#viewSelect"),
  results: document.querySelector("#results"),
  emptyState: document.querySelector("#emptyState"),
  visibleCount: document.querySelector("#visibleCount"),
  totalCount: document.querySelector("#totalCount"),
  datasetLabel: document.querySelector("#datasetLabel"),
  cardTemplate: document.querySelector("#cardTemplate"),
  chips: Array.from(document.querySelectorAll(".chip")),
  coverflowSection: document.querySelector("#coverflowSection"),
  coverflowViewport: document.querySelector("#coverflowViewport"),
  coverflowTrack: document.querySelector("#coverflowTrack"),
  coverflowMeta: document.querySelector("#coverflowMeta"),
  coverflowPrev: document.querySelector("#coverflowPrev"),
  coverflowNext: document.querySelector("#coverflowNext"),
};

const FALLBACK_AVATAR =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
    <defs>
      <linearGradient id="g" x1="0%" x2="100%" y1="0%" y2="100%">
        <stop offset="0%" stop-color="#efc47d"/>
        <stop offset="100%" stop-color="#cb5a32"/>
      </linearGradient>
    </defs>
    <rect width="400" height="400" rx="42" fill="url(#g)"/>
    <circle cx="200" cy="154" r="72" fill="rgba(255,255,255,.86)"/>
    <path d="M88 332c22-58 72-92 112-92s90 34 112 92" fill="rgba(255,255,255,.82)"/>
  </svg>
`);

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function loadDatasets() {
  const datasets = await fetchJson("/api/datasets");
  state.datasets = datasets;
  elements.datasetSelect.innerHTML = "";

  for (const dataset of datasets) {
    const option = document.createElement("option");
    option.value = dataset.file;
    option.textContent = `${dataset.name} (${dataset.count})`;
    elements.datasetSelect.append(option);
  }

  if (!datasets.length) {
    elements.datasetLabel.textContent = "No datasets";
    render();
    return;
  }

  await loadDataset(datasets[0].file);
}

async function loadDataset(file) {
  const dataset = state.datasets.find((item) => item.file === file);
  if (!dataset) {
    return;
  }

  state.activeDataset = dataset;
  state.records = await fetchJson(`/api/data?name=${encodeURIComponent(file)}`);
  elements.datasetSelect.value = file;
  elements.datasetLabel.textContent = dataset.name;
  elements.totalCount.textContent = String(state.records.length);
  render();
}

function normalizedText(record) {
  return [record.name, record.handle, record.description].join(" ").toLowerCase();
}

function filterRecords(records) {
  const query = elements.searchInput.value.trim().toLowerCase();
  const activeFilter = state.filter;

  return records.filter((record) => {
    const matchesQuery = !query || normalizedText(record).includes(query);
    const hasBio = Boolean(record.description?.trim());
    const hasImage = Boolean(record.profile_image_url?.trim());

    const matchesFilter =
      activeFilter === "all" ||
      (activeFilter === "bio" && hasBio) ||
      (activeFilter === "no-bio" && !hasBio) ||
      (activeFilter === "image" && hasImage);

    return matchesQuery && matchesFilter;
  });
}

function sortRecords(records) {
  const mode = elements.sortSelect.value;
  const sorted = [...records];
  const collator = new Intl.Collator(undefined, { sensitivity: "base" });

  sorted.sort((a, b) => {
    if (mode === "name-asc") return collator.compare(a.name || "", b.name || "");
    if (mode === "name-desc") return collator.compare(b.name || "", a.name || "");
    if (mode === "handle-asc") return collator.compare(a.handle || "", b.handle || "");
    if (mode === "handle-desc") return collator.compare(b.handle || "", a.handle || "");
    if (mode === "bio-desc") return (b.description || "").length - (a.description || "").length;
    if (mode === "bio-asc") return (a.description || "").length - (b.description || "").length;
    return 0;
  });

  return sorted;
}

function buildCard(record) {
  const fragment = elements.cardTemplate.content.cloneNode(true);
  const card = fragment.querySelector(".profile-card");
  const image = fragment.querySelector(".avatar");
  const name = fragment.querySelector(".name");
  const handle = fragment.querySelector(".handle");
  const bio = fragment.querySelector(".bio");
  const hoverName = fragment.querySelector(".hover-name");
  const hoverHandle = fragment.querySelector(".hover-handle");
  const hoverBio = fragment.querySelector(".hover-bio");
  const profileLink = fragment.querySelector(".profile-link");

  card.tabIndex = 0;
  image.src = record.profile_image_url || FALLBACK_AVATAR;
  image.alt = `${record.name || record.handle} profile picture`;
  image.addEventListener("error", () => {
    image.src = FALLBACK_AVATAR;
  });

  name.textContent = record.name || "Unknown";
  handle.textContent = record.handle || "";
  bio.textContent = record.description || "No bio provided.";
  hoverName.textContent = record.name || "Unknown";
  hoverHandle.textContent = record.handle || "";
  hoverBio.textContent = record.description || "No bio provided.";
  profileLink.href = record.profile_url || "#";

  return fragment;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function getCoverflowStep() {
  return window.innerWidth <= 720 ? 74 : 92;
}

function getCoverflowCardWidth() {
  return window.innerWidth <= 720 ? 168 : 220;
}

function buildCoverflowStructure() {
  const records = state.filteredRecords;
  const signature = records.map((record) => record.handle).join("|");
  if (state.coverflowSignature === signature) {
    return;
  }

  state.coverflowSignature = signature;
  elements.coverflowTrack.innerHTML = "";

  if (!records.length) {
    elements.coverflowMeta.innerHTML = "";
    return;
  }

  records.forEach((record, index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "coverflow-item";
    item.dataset.index = String(index);

    item.innerHTML = `
      <div class="coverflow-card">
        <div class="image-wrap">
          <img class="avatar" alt="${escapeHtml(record.name || record.handle)} profile picture" loading="lazy" src="${record.profile_image_url || FALLBACK_AVATAR}">
        </div>
      </div>
    `;

    const img = item.querySelector(".avatar");
    const coverCard = item.querySelector(".coverflow-card");
    coverCard.style.setProperty("--cover-reflection", `url("${record.profile_image_url || FALLBACK_AVATAR}")`);
    img.addEventListener("error", () => {
      img.src = FALLBACK_AVATAR;
      coverCard.style.setProperty("--cover-reflection", `url("${FALLBACK_AVATAR}")`);
    });

    item.addEventListener("click", () => {
      state.coverflowIndex = index;
      render();
    });

    elements.coverflowTrack.append(item);
  });
}

function updateCoverflow() {
  const records = state.filteredRecords;
  if (!records.length) {
    elements.coverflowMeta.innerHTML = "";
    return;
  }

  state.coverflowIndex = clamp(state.coverflowIndex, 0, records.length - 1);
  const step = getCoverflowStep();

  const items = Array.from(elements.coverflowTrack.children);
  for (const item of items) {
    const index = Number(item.dataset.index);
    const offset = index - state.coverflowIndex;
    const absOffset = Math.abs(offset);
    const rotateY = offset === 0 ? 0 : offset < 0 ? 62 : -62;
    const translateX = offset * step;
    const translateZ = offset === 0 ? 170 : Math.max(-240, 70 - absOffset * 64);
    const scale = offset === 0 ? 1.02 : Math.max(0.76, 0.95 - absOffset * 0.06);
    const opacity = absOffset > 6 ? 0 : Math.max(0.12, 1 - absOffset * 0.16);

    item.classList.toggle("is-active", offset === 0);
    item.style.transform =
      `translate3d(calc(-50% + ${translateX}px), 0, ${translateZ}px) rotateY(${rotateY}deg) scale(${scale})`;
    item.style.opacity = String(opacity);
    item.style.zIndex = String(500 - absOffset);
    item.style.filter = offset === 0 ? "none" : "saturate(0.72) brightness(0.72)";
    item.style.pointerEvents = absOffset > 7 ? "none" : "auto";
  }

  const active = records[state.coverflowIndex];
  elements.coverflowMeta.innerHTML = `
    <h2 class="name">${escapeHtml(active.name || "Unknown")}</h2>
    <p class="handle">${escapeHtml(active.handle || "")}</p>
    <p class="bio">${escapeHtml(active.description || "No bio provided.")}</p>
  `;
}

function render() {
  const filtered = sortRecords(filterRecords(state.records));
  const view = elements.viewSelect.value;
  state.filteredRecords = filtered;

  elements.results.className = `results-grid ${view}`;
  elements.results.innerHTML = "";
  elements.visibleCount.textContent = String(filtered.length);
  elements.emptyState.classList.toggle("hidden", filtered.length > 0);
  elements.coverflowSection.classList.toggle("hidden", view !== "coverflow" || filtered.length === 0);
  elements.results.style.display = view === "coverflow" ? "none" : "";

  if (view === "coverflow") {
    buildCoverflowStructure();
    updateCoverflow();
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const record of filtered) {
    fragment.append(buildCard(record));
  }
  elements.results.append(fragment);
}

function bindEvents() {
  elements.datasetSelect.addEventListener("change", (event) => {
    loadDataset(event.target.value);
  });

  elements.searchInput.addEventListener("input", render);
  elements.sortSelect.addEventListener("change", render);
  elements.viewSelect.addEventListener("change", render);
  elements.coverflowPrev.addEventListener("click", () => {
    state.coverflowIndex = clamp(state.coverflowIndex - 1, 0, state.filteredRecords.length - 1);
    render();
  });
  elements.coverflowNext.addEventListener("click", () => {
    state.coverflowIndex = clamp(state.coverflowIndex + 1, 0, state.filteredRecords.length - 1);
    render();
  });
  elements.coverflowViewport.addEventListener("wheel", (event) => {
    if (elements.viewSelect.value !== "coverflow" || !state.filteredRecords.length) {
      return;
    }
    event.preventDefault();
    const delta = event.deltaY || event.deltaX;
    if (Math.abs(delta) < 4) {
      return;
    }
    state.coverflowIndex = clamp(
      state.coverflowIndex + (delta > 0 ? 1 : -1),
      0,
      state.filteredRecords.length - 1,
    );
    render();
  }, { passive: false });
  elements.coverflowViewport.addEventListener("keydown", (event) => {
    if (elements.viewSelect.value !== "coverflow" || !state.filteredRecords.length) {
      return;
    }
    if (event.key === "ArrowLeft") {
      state.coverflowIndex = clamp(state.coverflowIndex - 1, 0, state.filteredRecords.length - 1);
      render();
    }
    if (event.key === "ArrowRight") {
      state.coverflowIndex = clamp(state.coverflowIndex + 1, 0, state.filteredRecords.length - 1);
      render();
    }
  });
  window.addEventListener("resize", () => {
    if (elements.viewSelect.value === "coverflow") {
      updateCoverflow();
    }
  });

  for (const chip of elements.chips) {
    chip.addEventListener("click", () => {
      for (const item of elements.chips) {
        item.classList.remove("active");
      }
      chip.classList.add("active");
      state.filter = chip.dataset.filter;
      render();
    });
  }
}

async function start() {
  bindEvents();
  try {
    await loadDatasets();
  } catch (error) {
    elements.datasetLabel.textContent = "Load failed";
    elements.emptyState.classList.remove("hidden");
    elements.emptyState.innerHTML = `<h2>Viewer failed to load</h2><p>${error.message}</p>`;
  }
}

start();
