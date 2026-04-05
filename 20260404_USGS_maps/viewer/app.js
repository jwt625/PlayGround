import maplibregl from "https://esm.sh/maplibre-gl@4.7.1";

const manifest = await fetch("./data/manifest.json").then((response) => response.json());

const state = {
  map: null,
  manifest,
  layers: new Map(),
  searchIndex: [],
  basemap: "light",
  theme: "light",
};

const lightTiles = [
  "https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
  "https://b.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
  "https://c.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
];

const satelliteTiles = [
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
];

const darkTiles = [
  "https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
  "https://b.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
  "https://c.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png",
];

const labelTiles = [
  "https://a.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
  "https://b.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
  "https://c.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
];

const darkLabelTiles = [
  "https://a.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png",
  "https://b.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png",
  "https://c.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png",
];

const statusChip = document.querySelector("#statusChip");
const layerList = document.querySelector("#layerList");
const legend = document.querySelector("#legend");
const detailsCard = document.querySelector("#detailsCard");
const selectionMeta = document.querySelector("#selectionMeta");
const searchInput = document.querySelector("#searchInput");
const searchResults = document.querySelector("#searchResults");
const sidebar = document.querySelector("#sidebar");
const appShell = document.querySelector(".app-shell");
const sidebarResizer = document.querySelector("#sidebarResizer");
const legendShell = document.querySelector("#legendShell");
const legendToggleButton = document.querySelector("#legendToggleButton");
const legendDragHandle = document.querySelector("#legendDragHandle");

document.body.dataset.theme = state.theme;
document.title = manifest.title;

setupMap();
renderLayerControls();
wireChrome();
updateLegend();
setupSidebarResize();
setupLegendInteractions();

function setupMap() {
  state.map = new maplibregl.Map({
    container: "map",
    center: manifest.defaultCenter,
    zoom: manifest.defaultZoom,
    minZoom: 2,
    maxZoom: 14,
    style: {
      version: 8,
      glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
      sources: {
        basemapLight: {
          type: "raster",
          tiles: lightTiles,
          tileSize: 256,
          attribution: "USGS dataset, Carto basemap",
        },
        basemapSatellite: {
          type: "raster",
          tiles: satelliteTiles,
          tileSize: 256,
        },
        labels: {
          type: "raster",
          tiles: labelTiles,
          tileSize: 256,
        },
      },
      layers: [
        { id: "basemap-light", type: "raster", source: "basemapLight" },
        {
          id: "basemap-satellite",
          type: "raster",
          source: "basemapSatellite",
          layout: { visibility: "none" },
        },
        { id: "labels", type: "raster", source: "labels", paint: { "raster-opacity": 0.85 } },
      ],
    },
  });

  state.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");
  state.map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
  state.map.addControl(new maplibregl.FullscreenControl(), "bottom-right");

  state.map.on("load", async () => {
    await Promise.all(
      manifest.layers
        .filter((layer) => layer.visibleByDefault)
        .map((layer) => ensureLayerLoaded(layer.id, true)),
    );
    statusChip.textContent = "Ready";
  });
}

function renderLayerControls() {
  const groups = groupBy(manifest.layers, "group");

  for (const [groupName, items] of Object.entries(groups)) {
    const group = document.createElement("div");
    group.className = "layer-group";
    group.dataset.group = groupName;
    group.innerHTML = `
      <button class="layer-group-header" type="button" data-group-toggle="${groupName}" aria-expanded="true">
        <div class="layer-group-title-wrap">
          <span class="layer-group-caret">▾</span>
          <span class="layer-group-title">${groupName}</span>
        </div>
        <div class="layer-group-actions">
          <button class="text-button" data-group-show="${groupName}">Show all</button>
          <button class="text-button" data-group-hide="${groupName}">Hide all</button>
        </div>
      </button>
      <div class="layer-group-body"></div>
    `;

    const groupBody = group.querySelector(".layer-group-body");

    for (const layer of items) {
      const wrapper = document.createElement("div");
      wrapper.className = "layer-item";
      wrapper.innerHTML = `
        <div class="layer-toggle">
          <div class="swatch" style="background:${layer.color}"></div>
          <div>
            <label class="layer-title">
              <input type="checkbox" data-layer-toggle="${layer.id}" ${layer.visibleByDefault ? "checked" : ""} />
              <span class="layer-name-row">
                <span>${layer.title}</span>
                <span class="layer-info" tabindex="0" aria-label="Layer description">
                  i
                  <span class="layer-tooltip">${escapeHtml(layer.summary)}</span>
                </span>
              </span>
            </label>
            <div class="layer-meta">${layer.featureCount.toLocaleString()} features · ${layer.geometryType}</div>
          </div>
          <button class="text-button" data-fit-layer="${layer.id}">Fit</button>
        </div>
        <div class="layer-actions">
          <input type="range" min="0.15" max="1" step="0.05" value="0.85" data-layer-opacity="${layer.id}" />
        </div>
      `;
      groupBody.appendChild(wrapper);
    }

    layerList.appendChild(group);
  }
}

function wireChrome() {
  document.querySelector("#homeButton").addEventListener("click", () => {
    state.map.flyTo({ center: manifest.defaultCenter, zoom: manifest.defaultZoom });
  });

  document.querySelector("#showAllButton").addEventListener("click", async () => {
    for (const layer of manifest.layers) {
      const checkbox = document.querySelector(`[data-layer-toggle="${layer.id}"]`);
      checkbox.checked = true;
      await ensureLayerLoaded(layer.id, true);
    }
    updateLegend();
  });

  document.querySelector("#hideAllButton").addEventListener("click", () => {
    for (const layer of manifest.layers) {
      const checkbox = document.querySelector(`[data-layer-toggle="${layer.id}"]`);
      checkbox.checked = false;
      setLayerVisibility(layer.id, false);
    }
    updateLegend();
  });

  document.querySelector("#basemapSwitcher").addEventListener("click", (event) => {
    const button = event.target.closest("[data-basemap]");
    if (!button) return;

    state.basemap = button.dataset.basemap;
    for (const item of document.querySelectorAll("#basemapSwitcher button")) {
      item.classList.toggle("is-active", item === button);
    }

    applyMapTheme();
  });

  document.querySelector("#themeSwitcher").addEventListener("click", (event) => {
    const button = event.target.closest("[data-theme]");
    if (!button) return;

    state.theme = button.dataset.theme;
    document.body.dataset.theme = state.theme;

    for (const item of document.querySelectorAll("#themeSwitcher button")) {
      item.classList.toggle("is-active", item === button);
    }

    applyMapTheme();
  });

  layerList.addEventListener("change", async (event) => {
    const checkbox = event.target.closest("[data-layer-toggle]");
    const opacity = event.target.closest("[data-layer-opacity]");

    if (checkbox) {
      await ensureLayerLoaded(checkbox.dataset.layerToggle, checkbox.checked);
      updateLegend();
    }

    if (opacity) {
      setLayerOpacity(opacity.dataset.layerOpacity, Number(opacity.value));
    }
  });

  layerList.addEventListener("click", async (event) => {
    const groupToggle = event.target.closest("[data-group-toggle]");
    if (groupToggle && !event.target.closest("[data-group-show], [data-group-hide]")) {
      const group = groupToggle.closest(".layer-group");
      const collapsed = group.classList.toggle("is-collapsed");
      groupToggle.setAttribute("aria-expanded", String(!collapsed));
      return;
    }

    const fitButton = event.target.closest("[data-fit-layer]");
    if (fitButton) {
      await ensureLayerLoaded(fitButton.dataset.fitLayer, true);
      fitToLayer(fitButton.dataset.fitLayer);
      return;
    }

    const groupShowButton = event.target.closest("[data-group-show]");
    if (groupShowButton) {
      const groupName = groupShowButton.dataset.groupShow;
      const layers = manifest.layers.filter((layer) => layer.group === groupName);
      for (const layer of layers) {
        const checkbox = document.querySelector(`[data-layer-toggle="${layer.id}"]`);
        checkbox.checked = true;
        await ensureLayerLoaded(layer.id, true);
      }
      updateLegend();
      return;
    }

    const groupHideButton = event.target.closest("[data-group-hide]");
    if (groupHideButton) {
      const groupName = groupHideButton.dataset.groupHide;
      const layers = manifest.layers.filter((layer) => layer.group === groupName);
      for (const layer of layers) {
        const checkbox = document.querySelector(`[data-layer-toggle="${layer.id}"]`);
        checkbox.checked = false;
        setLayerVisibility(layer.id, false);
      }
      updateLegend();
    }
  });

  searchInput.addEventListener("input", () => runSearch(searchInput.value));
  document.querySelector("#openSidebarButton").addEventListener("click", () => sidebar.classList.add("is-open"));
  document.querySelector("#closeSidebarButton").addEventListener("click", () => sidebar.classList.remove("is-open"));
  legendToggleButton.addEventListener("click", () => {
    const expanded = legendShell.classList.toggle("is-expanded");
    legendToggleButton.textContent = expanded ? "Fold" : "Unfold";
  });
}

async function ensureLayerLoaded(layerId, visible) {
  const existing = state.layers.get(layerId);
  if (existing) {
    setLayerVisibility(layerId, visible);
    return;
  }
  if (!visible) return;

  const descriptor = manifest.layers.find((layer) => layer.id === layerId);
  if (!descriptor) return;

  statusChip.textContent = `Loading ${descriptor.title}…`;
  const data = await fetch(descriptor.source).then((response) => response.json());
  const sourceId = `${layerId}-source`;
  const paintOpacity = 0.85;

  state.map.addSource(sourceId, {
    type: "geojson",
    data,
    cluster: descriptor.geometryType === "Point" && descriptor.featureCount > 250,
    clusterRadius: 42,
  });

  if (descriptor.geometryType === "Point") {
    if (descriptor.featureCount > 250) {
      state.map.addLayer({
        id: `${layerId}-clusters`,
        type: "circle",
        source: sourceId,
        filter: ["has", "point_count"],
        paint: {
          "circle-color": descriptor.color,
          "circle-radius": ["step", ["get", "point_count"], 14, 20, 18, 80, 24],
          "circle-opacity": 0.8,
        },
      });
      state.map.addLayer({
        id: `${layerId}-cluster-count`,
        type: "symbol",
        source: sourceId,
        filter: ["has", "point_count"],
        layout: { "text-field": ["get", "point_count_abbreviated"], "text-size": 11 },
        paint: { "text-color": "#ffffff" },
      });
    }

    const pointLayer = {
      id: `${layerId}-points`,
      type: "circle",
      source: sourceId,
      paint: {
        "circle-color": descriptor.color,
        "circle-radius": 4.5,
        "circle-stroke-width": 1,
        "circle-stroke-color": "#f8fafc",
        "circle-opacity": paintOpacity,
      },
    };

    if (descriptor.featureCount > 250) {
      pointLayer.filter = ["!", ["has", "point_count"]];
    }

    state.map.addLayer(pointLayer);
  } else if (descriptor.geometryType.includes("Line")) {
    state.map.addLayer({
      id: `${layerId}-lines`,
      type: "line",
      source: sourceId,
      paint: {
        "line-color": descriptor.color,
        "line-width": 2.2,
        "line-opacity": paintOpacity,
      },
    });
  } else {
    state.map.addLayer({
      id: `${layerId}-fill`,
      type: "fill",
      source: sourceId,
      paint: {
        "fill-color": descriptor.color,
        "fill-opacity": 0.22,
      },
    });
    state.map.addLayer({
      id: `${layerId}-outline`,
      type: "line",
      source: sourceId,
      paint: {
        "line-color": descriptor.color,
        "line-width": 1.2,
        "line-opacity": paintOpacity,
      },
    });
  }

  wireLayerInteractions(descriptor, sourceId, data);
  state.layers.set(layerId, { descriptor, data, sourceId });
  setLayerVisibility(layerId, visible);
  buildSearchIndex();
  statusChip.textContent = "Ready";
}

function wireLayerInteractions(descriptor, sourceId, data) {
  const interactiveIds = getInteractiveLayerIds(descriptor.id);
  const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 14 });

  for (const layerId of interactiveIds) {
    if (!state.map.getLayer(layerId)) continue;

    state.map.on("mouseenter", layerId, () => {
      state.map.getCanvas().style.cursor = "pointer";
    });

    state.map.on("mouseleave", layerId, () => {
      state.map.getCanvas().style.cursor = "";
      popup.remove();
    });

    state.map.on("mousemove", layerId, (event) => {
      const feature = event.features?.[0];
      if (!feature) return;
      const isCluster = Boolean(feature.properties?.point_count);
      const title = isCluster
        ? `${feature.properties.point_count.toLocaleString()} features`
        : pickDisplayText(feature.properties?.title, descriptor.title);
      const subtitle = isCluster
        ? descriptor.title
        : pickDisplayText(feature.properties?.subtitle, descriptor.summary, descriptor.title);
      popup
        .setLngLat(event.lngLat)
        .setHTML(
          `<div class="popup-card"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(subtitle)}</span></div>`,
        )
        .addTo(state.map);
    });

    state.map.on("click", layerId, (event) => {
      const feature = event.features?.[0];
      if (!feature) return;

      if (feature.properties.cluster) return;

      if (descriptor.geometryType === "Point" && feature.properties.point_count) {
        state.map.getSource(sourceId).getClusterExpansionZoom(feature.properties.cluster_id, (error, zoom) => {
          if (error) return;
          state.map.easeTo({ center: feature.geometry.coordinates, zoom });
        });
        return;
      }

      renderDetails(feature.properties, descriptor);
    });
  }
}

function applyMapTheme() {
  const lightVisible = state.basemap === "light";

  state.map.getSource("basemapLight").setTiles(state.theme === "dark" ? darkTiles : lightTiles);
  state.map.getSource("labels").setTiles(state.theme === "dark" ? darkLabelTiles : labelTiles);

  state.map.setLayoutProperty("basemap-light", "visibility", lightVisible ? "visible" : "none");
  state.map.setLayoutProperty(
    "basemap-satellite",
    "visibility",
    state.basemap === "satellite" ? "visible" : "none",
  );
  state.map.setPaintProperty("labels", "raster-opacity", state.basemap === "satellite" ? 1 : 0.85);
}

function renderDetails(properties, descriptor) {
  const ignored = new Set(["title", "subtitle", "searchText"]);
  const rows = Object.entries(properties)
    .filter(([key, value]) => normalizeValue(value) && !ignored.has(key))
    .slice(0, 14)
    .map(
      ([key, value]) => `
        <div class="detail-row">
          <div class="detail-key">${formatLabel(key)}</div>
          <div class="detail-value">${escapeHtml(normalizeValue(value))}</div>
        </div>
      `,
    )
    .join("");

  selectionMeta.textContent = `${descriptor.region} · ${descriptor.title}`;
  detailsCard.classList.remove("empty");
  detailsCard.innerHTML = `
    <strong>${escapeHtml(pickDisplayText(properties.title, descriptor.title))}</strong>
    <span>${escapeHtml(pickDisplayText(properties.subtitle, descriptor.summary, descriptor.title))}</span>
    ${rows}
  `;
}

function setLayerVisibility(layerId, visible) {
  for (const id of getInteractiveLayerIds(layerId)) {
    if (state.map.getLayer(id)) {
      state.map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
    }
  }
}

function setLayerOpacity(layerId, value) {
  const descriptor = manifest.layers.find((layer) => layer.id === layerId);
  if (!descriptor) return;

  if (descriptor.geometryType === "Point") {
    if (state.map.getLayer(`${layerId}-points`)) {
      state.map.setPaintProperty(`${layerId}-points`, "circle-opacity", value);
    }
    if (state.map.getLayer(`${layerId}-clusters`)) {
      state.map.setPaintProperty(`${layerId}-clusters`, "circle-opacity", Math.min(1, value + 0.1));
    }
  } else if (descriptor.geometryType.includes("Line")) {
    if (state.map.getLayer(`${layerId}-lines`)) {
      state.map.setPaintProperty(`${layerId}-lines`, "line-opacity", value);
    }
  } else {
    if (state.map.getLayer(`${layerId}-fill`)) {
      state.map.setPaintProperty(`${layerId}-fill`, "fill-opacity", value * 0.35);
    }
    if (state.map.getLayer(`${layerId}-outline`)) {
      state.map.setPaintProperty(`${layerId}-outline`, "line-opacity", value);
    }
  }
}

function fitToLayer(layerId) {
  const entry = state.layers.get(layerId);
  if (!entry) return;
  const bounds = computeBounds(entry.data);
  if (!bounds) return;
  state.map.fitBounds(bounds, { padding: 48, duration: 700 });
}

function updateLegend() {
  const visible = manifest.layers.filter((layer) => {
    const checkbox = document.querySelector(`[data-layer-toggle="${layer.id}"]`);
    return checkbox?.checked;
  });

  if (!visible.length) {
    legend.innerHTML = "<div class='legend-item'><span>All thematic layers are hidden.</span></div>";
    return;
  }

  legend.innerHTML = `${visible
    .map((layer) => {
      const marker =
        layer.geometryType === "Point"
          ? `<span class="swatch" style="background:${layer.color}"></span>`
          : layer.geometryType.includes("Line")
            ? `<span class="legend-line" style="background:${layer.color}"></span>`
            : `<span class="legend-fill" style="background:${layer.color}"></span>`;
      return `<div class="legend-item">${marker}<span>${layer.region} · ${layer.title}</span></div>`;
    })
    .join("")}`;
}

function buildSearchIndex() {
  state.searchIndex = [];
  for (const entry of state.layers.values()) {
    const enabled = document.querySelector(`[data-layer-toggle="${entry.descriptor.id}"]`)?.checked;
    if (!enabled) continue;
    for (const feature of entry.data.features.slice(0, 500)) {
      state.searchIndex.push({
        layerId: entry.descriptor.id,
        title: feature.properties.title || entry.descriptor.title,
        subtitle: [entry.descriptor.region, feature.properties.subtitle || entry.descriptor.title]
          .filter(Boolean)
          .join(" · "),
        searchText: feature.properties.searchText || "",
        geometry: feature.geometry,
      });
    }
  }
}

function runSearch(query) {
  const term = query.trim().toLowerCase();
  searchResults.innerHTML = "";

  if (!term) return;

  const matches = state.searchIndex
    .filter((item) => item.searchText.toLowerCase().includes(term))
    .slice(0, 12);

  if (!matches.length) {
    searchResults.innerHTML = `<div class="search-result"><strong>No matches</strong><span>Try another place, commodity, or operator.</span></div>`;
    return;
  }

  for (const match of matches) {
    const button = document.createElement("button");
    button.className = "search-result";
    button.innerHTML = `<strong>${escapeHtml(pickDisplayText(match.title, "Untitled feature"))}</strong><span>${escapeHtml(pickDisplayText(match.subtitle, "No secondary label"))}</span>`;
    button.addEventListener("click", () => {
      const bounds = computeBounds({ type: "FeatureCollection", features: [{ geometry: match.geometry, properties: {} }] });
      if (bounds) {
        state.map.fitBounds(bounds, { padding: 64, maxZoom: 8, duration: 700 });
      }
      sidebar.classList.remove("is-open");
    });
    searchResults.appendChild(button);
  }
}

function computeBounds(geojson) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const feature of geojson.features) {
    visitCoordinates(feature.geometry?.coordinates, (coord) => {
      const [x, y] = coord;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    });
  }

  if (!Number.isFinite(minX)) return null;
  return [
    [minX, minY],
    [maxX, maxY],
  ];
}

function visitCoordinates(coordinates, callback) {
  if (!Array.isArray(coordinates)) return;
  if (typeof coordinates[0] === "number") {
    callback(coordinates);
    return;
  }
  for (const child of coordinates) {
    visitCoordinates(child, callback);
  }
}

function getInteractiveLayerIds(layerId) {
  return [`${layerId}-clusters`, `${layerId}-cluster-count`, `${layerId}-points`, `${layerId}-lines`, `${layerId}-fill`, `${layerId}-outline`];
}

function groupBy(items, key) {
  return items.reduce((accumulator, item) => {
    accumulator[item[key]] ||= [];
    accumulator[item[key]].push(item);
    return accumulator;
  }, {});
}

function formatLabel(value) {
  return value.replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2");
}

function setupSidebarResize() {
  if (!sidebarResizer) return;

  sidebarResizer.addEventListener("pointerdown", (event) => {
    if (window.innerWidth <= 960) return;

    const startX = event.clientX;
    const startWidth = sidebar.getBoundingClientRect().width;
    sidebarResizer.setPointerCapture(event.pointerId);

    const onMove = (moveEvent) => {
      const nextWidth = Math.min(520, Math.max(260, startWidth + moveEvent.clientX - startX));
      appShell.style.setProperty("--sidebar-width", `${nextWidth}px`);
    };

    const onUp = () => {
      sidebarResizer.removeEventListener("pointermove", onMove);
      sidebarResizer.removeEventListener("pointerup", onUp);
      sidebarResizer.removeEventListener("pointercancel", onUp);
    };

    sidebarResizer.addEventListener("pointermove", onMove);
    sidebarResizer.addEventListener("pointerup", onUp);
    sidebarResizer.addEventListener("pointercancel", onUp);
  });
}

function setupLegendInteractions() {
  if (!legendShell || !legendDragHandle) return;

  legendDragHandle.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button")) return;

    const parent = legendShell.offsetParent || legendShell.parentElement;
    const parentRect = parent.getBoundingClientRect();
    const rect = legendShell.getBoundingClientRect();
    const pointerOffsetX = event.clientX - rect.left;
    const pointerOffsetY = event.clientY - rect.top;
    const startLeft = rect.left - parentRect.left;
    const startTop = rect.top - parentRect.top;

    legendShell.style.left = `${startLeft}px`;
    legendShell.style.top = `${startTop}px`;
    legendShell.style.right = "auto";
    legendShell.style.bottom = "auto";

    legendDragHandle.setPointerCapture(event.pointerId);

    const onMove = (moveEvent) => {
      const maxLeft = Math.max(8, parentRect.width - rect.width - 8);
      const maxTop = Math.max(8, parentRect.height - legendShell.offsetHeight - 8);
      const nextLeft = moveEvent.clientX - parentRect.left - pointerOffsetX;
      const nextTop = moveEvent.clientY - parentRect.top - pointerOffsetY;
      legendShell.style.left = `${Math.min(maxLeft, Math.max(8, nextLeft))}px`;
      legendShell.style.top = `${Math.min(maxTop, Math.max(8, nextTop))}px`;
    };

    const onUp = () => {
      legendDragHandle.removeEventListener("pointermove", onMove);
      legendDragHandle.removeEventListener("pointerup", onUp);
      legendDragHandle.removeEventListener("pointercancel", onUp);
    };

    legendDragHandle.addEventListener("pointermove", onMove);
    legendDragHandle.addEventListener("pointerup", onUp);
    legendDragHandle.addEventListener("pointercancel", onUp);
  });
}

function normalizeValue(value) {
  if (value === null || value === undefined) return "";
  const text = String(value).trim();
  if (!text || text === "undefined" || text === "null" || text === "NaN") return "";
  return text;
}

function pickDisplayText(...values) {
  for (const value of values) {
    const normalized = normalizeValue(value);
    if (normalized) return normalized;
  }
  return "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
