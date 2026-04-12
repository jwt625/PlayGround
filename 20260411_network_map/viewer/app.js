const state = {
  datasets: [],
  activeDataset: null,
  records: [],
  filter: "all",
  filteredRecords: [],
  coverflowIndex: 0,
  coverflowSignature: "",
  pieLegendSort: "fraction-desc",
  statsValueDisplay: "fraction-only",
};

const ROLE_BUCKETS = [
  {
    key: "no_bio",
    label: "No Bio",
    color: "#9f988b",
    match(record) {
      return !record.description?.trim();
    },
  },
  {
    key: "founder",
    label: "Founder",
    color: "#cb5a32",
    patterns: [/\bco-?founder\b/, /\bfounder\b/, /\bbuilding\b/, /\bstarted\b/],
  },
  {
    key: "vc",
    label: "VC / Investor",
    color: "#d8873f",
    patterns: [
      /\bventure capitalist\b/,
      /\bvc\b/,
      /\binvestor\b/,
      /\bangel investor\b/,
      /\bseed investor\b/,
      /\bgp\b/,
      /\blp\b/,
      /\bpartner\b/,
      /\bpartners\b/,
      /\bgeneral partner\b/,
      /\blimited partner\b/,
      /\bmanaging partner\b/,
    ],
  },
  {
    key: "ceo",
    label: "CEO",
    color: "#efc47d",
    patterns: [/\bceo\b/, /\bchief executive officer\b/],
  },
  {
    key: "cto",
    label: "CTO",
    color: "#8fb996",
    patterns: [/\bcto\b/, /\bchief technology officer\b/],
  },
  {
    key: "swe",
    label: "SWE",
    color: "#5aa9b2",
    patterns: [
      /\bsoftware engineer\b/,
      /\bswe\b/,
      /\bsoftware developer\b/,
      /\bdeveloper\b/,
      /\bfull stack\b/,
      /\bbackend\b/,
      /\bfrontend\b/,
      /\bsre\b/,
      /\bprogrammer\b/,
      /\bcoder\b/,
    ],
  },
  {
    key: "ai_ml",
    label: "AI / ML",
    color: "#4d7ea8",
    patterns: [
      /\bartificial intelligence\b/,
      /\bmachine learning\b/,
      /\bai\b/,
      /\bml\b/,
      /\blanguage model\b/,
      /\blanguage models\b/,
      /\bllm\b/,
      /\bgenai\b/,
      /\bdeep learning\b/,
      /\binference\b/,
    ],
  },
  {
    key: "robotics",
    label: "Robotics",
    color: "#6d597a",
    patterns: [/\brobotics\b/, /\brobots\b/, /\brobot\b/, /\bautonomy\b/, /\bdriverless\b/, /\bdrone\b/],
  },
  {
    key: "chip_nanofab",
    label: "Chip / Nanofab",
    color: "#a16ae8",
    patterns: [
      /\bsemiconductor\b/,
      /\bchip\b/,
      /\bchiplets\b/,
      /\basic\b/,
      /\bvlsi\b/,
      /\bfpga\b/,
      /\bpcb\b/,
      /\bnanofab\b/,
      /\bnanofabrication\b/,
      /\bprocess node\b/,
      /\bsilicon\b/,
      /\brisc-v\b/,
      /\bembedded\b/,
    ],
  },
  {
    key: "electronics",
    label: "Electronics",
    color: "#8d99ae",
    patterns: [
      /\belectronics\b/,
      /\belectronic\b/,
      /\belectronics engineer\b/,
      /\belectrical engineer\b/,
      /\belectrical engineering\b/,
      /\bcircuit\b/,
      /\bcircuits\b/,
      /\banalog\b/,
      /\bdigital design\b/,
      /\bpcb\b/,
      /\bembedded systems\b/,
      /\bfirmware\b/,
    ],
  },
  {
    key: "optics_photonics_rf",
    label: "Optics / Photonics / RF",
    color: "#7b9acc",
    patterns: [
      /\boptics\b/,
      /\boptical\b/,
      /\bphotonics\b/,
      /\bphotonic\b/,
      /\brf\b/,
      /\bradio frequency\b/,
      /\bmicrowave\b/,
      /\bantenna\b/,
      /\blidar\b/,
      /\blasers?\b/,
    ],
  },
  {
    key: "research_scientist",
    label: "Research / Scientist",
    color: "#8a6f52",
    patterns: [
      /\bresearcher\b/,
      /\bresearch\b/,
      /\bscientist\b/,
      /\bscience\b/,
      /\bphysicist\b/,
      /\bphysics\b/,
      /\bbiophysics\b/,
      /\bmathematician\b/,
      /\bmath\b/,
      /\bcryptology\b/,
      /\bosint\b/,
    ],
  },
  {
    key: "head",
    label: "Head / Lead",
    color: "#84a59d",
    patterns: [/\bhead of\b/, /\bhead\b/, /\blead\b/, /\bvp\b/, /\bvice president\b/, /\bdirector\b/],
  },
  {
    key: "professor",
    label: "Professor",
    color: "#52796f",
    patterns: [/\bprofessor\b/, /\bassociate professor\b/, /\bassistant professor\b/, /\blecturer\b/, /\bfaculty\b/],
  },
  {
    key: "phd",
    label: "PhD",
    color: "#b56576",
    patterns: [/\bphd\b/, /\bph\.d\b/, /\bdoctoral\b/, /\bdoctorate\b/],
  },
  {
    key: "masters",
    label: "Master's",
    color: "#355070",
    patterns: [/\bmasters\b/, /\bmaster's\b/, /\bm\.sc\b/, /\bmsc\b/, /\bms\b/],
  },
  {
    key: "undergrad",
    label: "Undergrad",
    color: "#6c757d",
    patterns: [
      /\bundergrad\b/,
      /\bundergraduate\b/,
      /\bcollege student\b/,
      /\bcs student\b/,
      /\bee student\b/,
      /\bme student\b/,
      /\bstudent @\b/,
      /\bstudying at\b/,
      /\bclass of 20\d{2}\b/,
      /\bb\.s\b/,
      /\bbs\b/,
      /\bba\b/,
    ],
  },
];

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
  profileModal: document.querySelector("#profileModal"),
  modalAvatar: document.querySelector("#modalAvatar"),
  modalName: document.querySelector("#modalName"),
  modalHandle: document.querySelector("#modalHandle"),
  modalBio: document.querySelector("#modalBio"),
  modalProfileLink: document.querySelector("#modalProfileLink"),
  modalImageLink: document.querySelector("#modalImageLink"),
  statsVisibleCount: document.querySelector("#statsVisibleCount"),
  statsMatchedCount: document.querySelector("#statsMatchedCount"),
  roleBars: document.querySelector("#roleBars"),
  rolePie: document.querySelector("#rolePie"),
  pieCenterValue: document.querySelector("#pieCenterValue"),
  pieLegend: document.querySelector("#pieLegend"),
  pieLegendSortSelect: document.querySelector("#pieLegendSortSelect"),
  statsValueDisplaySelect: document.querySelector("#statsValueDisplaySelect"),
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

function getRoleMatches(record) {
  const haystack = normalizedText(record);
  return ROLE_BUCKETS.filter((bucket) => {
    if (typeof bucket.match === "function") {
      return bucket.match(record);
    }
    return bucket.patterns.some((pattern) => pattern.test(haystack));
  });
}

function computeRoleStats(records) {
  const overlappingCounts = Object.fromEntries(ROLE_BUCKETS.map((bucket) => [bucket.key, 0]));
  const pieCounts = Object.fromEntries(ROLE_BUCKETS.map((bucket) => [bucket.key, 0]));
  let matchedProfiles = 0;

  for (const record of records) {
    const matches = getRoleMatches(record);
    if (matches.length) {
      matchedProfiles += 1;
    }
    for (const bucket of matches) {
      overlappingCounts[bucket.key] += 1;
    }
    const primaryBucket = matches[0]?.key || "other";
    if (primaryBucket === "other") {
      continue;
    }
    pieCounts[primaryBucket] += 1;
  }

  const barRows = ROLE_BUCKETS.map((bucket) => {
    const count = overlappingCounts[bucket.key];
    return {
      ...bucket,
      count,
      ratio: records.length ? count / records.length : 0,
    };
  }).sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));

  const pieRows = ROLE_BUCKETS.map((bucket) => ({
    ...bucket,
    count: pieCounts[bucket.key],
  })).filter((row) => row.count > 0);

  const unmatchedCount = records.length - pieRows.reduce((sum, row) => sum + row.count, 0);
  if (unmatchedCount > 0) {
    pieRows.push({
      key: "other",
      label: "Other",
      color: "#c7c1b1",
      count: unmatchedCount,
    });
  }

  return {
    matchedProfiles,
    totalProfiles: records.length,
    barRows,
    pieRows,
  };
}

function formatPercent(value) {
  return `${(value * 100).toFixed(value > 0 && value < 0.1 ? 1 : 0)}%`;
}

function formatStatValue(count, ratio) {
  const fraction = formatPercent(ratio);
  if (state.statsValueDisplay === "fraction-only") {
    return fraction;
  }
  return `${count} · ${fraction}`;
}

function sortPieRows(rows, totalProfiles) {
  const mode = state.pieLegendSort;
  const collator = new Intl.Collator(undefined, { sensitivity: "base" });
  const sorted = [...rows];

  sorted.sort((a, b) => {
    const aRatio = totalProfiles ? a.count / totalProfiles : 0;
    const bRatio = totalProfiles ? b.count / totalProfiles : 0;

    if (mode === "fraction-desc") {
      return bRatio - aRatio || collator.compare(a.label, b.label);
    }
    if (mode === "fraction-asc") {
      return aRatio - bRatio || collator.compare(a.label, b.label);
    }
    if (mode === "label-desc") {
      return collator.compare(b.label, a.label);
    }
    return collator.compare(a.label, b.label);
  });

  return sorted;
}

function renderStats(records) {
  const stats = computeRoleStats(records);
  elements.statsVisibleCount.textContent = String(stats.totalProfiles);
  elements.statsMatchedCount.textContent = `${stats.matchedProfiles} (${formatPercent(
    stats.totalProfiles ? stats.matchedProfiles / stats.totalProfiles : 0,
  )})`;
  elements.pieCenterValue.textContent = String(stats.totalProfiles);

  elements.roleBars.innerHTML = "";
  const barFragment = document.createDocumentFragment();
  for (const row of stats.barRows) {
    const item = document.createElement("div");
    item.className = "role-bar-row";
    item.innerHTML = `
      <div class="role-bar-meta">
        <span class="role-bar-label">${escapeHtml(row.label)}</span>
        <span class="role-bar-value">${formatStatValue(row.count, row.ratio)}</span>
      </div>
      <div class="role-bar-track">
        <div class="role-bar-fill" style="--bar-width: ${row.ratio * 100}%; --bar-color: ${row.color};"></div>
      </div>
    `;
    barFragment.append(item);
  }
  elements.roleBars.append(barFragment);

  if (!stats.pieRows.length) {
    elements.rolePie.style.background = "conic-gradient(#e6decd 0turn 1turn)";
    elements.pieLegend.innerHTML = `<p class="pie-empty">No visible profiles to summarize.</p>`;
    return;
  }

  let current = 0;
  const stops = [];
  for (const row of stats.pieRows) {
    const slice = stats.totalProfiles ? row.count / stats.totalProfiles : 0;
    const next = current + slice;
    stops.push(`${row.color} ${current}turn ${next}turn`);
    current = next;
  }
  elements.rolePie.style.background = `conic-gradient(${stops.join(", ")})`;
  const legendRows = sortPieRows(stats.pieRows, stats.totalProfiles);
  elements.pieLegend.innerHTML = legendRows.map((row) => {
    const ratio = stats.totalProfiles ? row.count / stats.totalProfiles : 0;
    return `
      <div class="pie-legend-row">
        <span class="legend-swatch" style="--swatch-color: ${row.color};"></span>
        <span class="legend-label">${escapeHtml(row.label)}</span>
        <span class="legend-value">${formatStatValue(row.count, ratio)}</span>
      </div>
    `;
  }).join("");
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
  card.dataset.profileUrl = record.profile_url || "";
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
  card.addEventListener("click", (event) => {
    if (shouldIgnoreCardActivation(event)) {
      return;
    }
    openProfileModal(record);
  });
  card.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    if (shouldIgnoreCardActivation(event)) {
      return;
    }
    event.preventDefault();
    openProfileModal(record);
  });

  return fragment;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function hasSelectedText() {
  const selection = window.getSelection();
  return Boolean(selection && selection.toString().trim());
}

function shouldIgnoreCardActivation(event) {
  const interactiveTarget = event.target.closest("a, button, input, select, textarea");
  return Boolean(interactiveTarget) || hasSelectedText();
}

function openProfileModal(record) {
  elements.modalAvatar.src = record.profile_image_url || FALLBACK_AVATAR;
  elements.modalAvatar.alt = `${record.name || record.handle || "Unknown"} profile picture`;
  elements.modalName.textContent = record.name || "Unknown";
  elements.modalHandle.textContent = record.handle || "";
  elements.modalBio.textContent = record.description || "No bio provided.";
  elements.modalProfileLink.href = record.profile_url || "#";
  elements.modalProfileLink.setAttribute("aria-disabled", record.profile_url ? "false" : "true");
  elements.modalImageLink.href = record.profile_image_url || "#";
  elements.modalImageLink.setAttribute("aria-disabled", record.profile_image_url ? "false" : "true");
  elements.profileModal.showModal();
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
      openProfileModal(record);
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

  renderStats(filtered);
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
  elements.pieLegendSortSelect.addEventListener("change", (event) => {
    state.pieLegendSort = event.target.value;
    render();
  });
  elements.statsValueDisplaySelect.addEventListener("change", (event) => {
    state.statsValueDisplay = event.target.value;
    render();
  });
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
  elements.profileModal.addEventListener("click", (event) => {
    const bounds = elements.profileModal.getBoundingClientRect();
    const isBackdropClick =
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom;
    if (isBackdropClick) {
      elements.profileModal.close();
    }
  });
  elements.modalAvatar.addEventListener("error", () => {
    elements.modalAvatar.src = FALLBACK_AVATAR;
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
