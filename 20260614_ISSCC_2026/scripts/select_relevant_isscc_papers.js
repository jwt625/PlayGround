const fs = require("fs");
const path = require("path");

const inputPath = path.resolve("cache/isscc_2026_ieee_xplore_papers.json");
const outputPath = path.resolve("cache/isscc_2026_relevant_papers.json");

const selection = {
  "11409081": {
    priority: "A",
    categories: ["optical_io", "dwDM", "clock_forwarding", "stacked_eic_pic"],
    thesis_links: ["stacked_low_baud_dwdm_tfln_pic", "timing_spine", "wdm_lane_rate_tradeoff"],
    rationale:
      "Closest ISSCC hit to the stacked low-baud DWDM architecture idea: 32 Gb/s/lambda, 256 Gb/s/fiber, forwarded clocking, and 3D-stacked EIC/PIC integration.",
  },
  "11409067": {
    priority: "A",
    categories: ["photonic_interposer", "electro_optical_router", "ocs", "low_latency"],
    thesis_links: ["scheduled_ocs", "optical_memory_fabric", "timing_spine"],
    rationale:
      "Directly relevant to scheduled optical fabrics: electro-optical router, 18 ns setup, frame-level routing, wavelength-flexible capacity, and photonic interposer context.",
  },
  "11409254": {
    priority: "A",
    categories: ["optical_io", "wdm_transmitter", "low_energy_tx", "cmos_siph"],
    thesis_links: ["wdm_lane_rate_tradeoff", "low_swing_optical_bus"],
    rationale:
      "WDM transmitter in CMOS SOI with explicit fJ/bit metric; useful comparator for TFLN low-electronics-burden optical I/O.",
  },
  "11409176": {
    priority: "A",
    categories: ["coherent_lite", "optical_transceiver", "latency", "800g"],
    thesis_links: ["scale_across", "coherent_lite", "serdes_dsp_tax"],
    rationale:
      "Coherent-lite 800 Gb/s transceiver with sub-300 ns latency; important benchmark for long-reach/scheduled capacity and latency-class optical links.",
  },
  "11409086": {
    priority: "A",
    categories: ["silicon_photonics", "dwdm", "pam4_transceiver", "monolithic"],
    thesis_links: ["wdm_lane_rate_tradeoff", "optical_io_competitive_map"],
    rationale:
      "Monolithic silicon-photonic DWDM PAM4 transceiver at 2x500 Gb/s; key SiPh comparator against TFLN-on-SiPh optical engine claims.",
  },
  "11409120": {
    priority: "A",
    categories: ["ucie", "die_to_die", "chiplet_io", "package_reach"],
    thesis_links: ["optical_memory_demo", "package_beachfront", "electrical_baseline"],
    rationale:
      "UCIe-compliant 48 Gb/s/lane D2D over standard package; strong electrical baseline for when optics must beat package-level copper.",
  },
  "11409144": {
    priority: "A",
    categories: ["ucie_like", "die_to_die", "energy_per_bit", "advanced_package"],
    thesis_links: ["optical_memory_demo", "low_energy_io_baseline"],
    rationale:
      "0.36 pJ/b UCIe-like D2D interface with high bandwidth density; sets a difficult short-reach electrical energy benchmark.",
  },
  "11409276": {
    priority: "A",
    categories: ["die_to_die", "modular_io", "clock_gating", "energy_per_bit"],
    thesis_links: ["low_power_io", "electrical_baseline", "timing_spine"],
    rationale:
      "0.23 pJ/b 24 Gb/s modular D2D interface; relevant to the low-baud/many-lane alternative and wake/acquisition overhead.",
  },
  "11409095": {
    priority: "A",
    categories: ["112g", "single_ended", "d2d", "equalization"],
    thesis_links: ["serdes_dsp_tax", "package_channel_limits"],
    rationale:
      "112 Gb/s/wire single-ended bidirectional transceiver; useful for electrical channel, equalizer, and package-limit comparison.",
  },
  "11409236": {
    priority: "A",
    categories: ["112g", "cdr", "pam4", "mixed_signal"],
    thesis_links: ["timing_spine", "cdr_lite", "serdes_dsp_tax"],
    rationale:
      "Reference-less mixed-signal PAM4 CDR at 112 Gb/s; directly informs the timing/CDR burden that a shared timing or CDR-lite architecture would need to reduce or amortize.",
  },
  "11409174": {
    priority: "A",
    categories: ["112g", "pam4", "nrz", "low_power_io"],
    thesis_links: ["low_swing_optical_bus", "serdes_baseline"],
    rationale:
      "112 Gb/s PAM4/NRZ low-power I/O transceiver; relevant baseline for NRZ-vs-PAM and low-power endpoint design.",
  },
  "11408972": {
    priority: "A",
    categories: ["240g", "pam4_tx", "analog_intensive", "energy_per_bit"],
    thesis_links: ["serdes_dsp_tax", "lane_rate_tradeoff"],
    rationale:
      "180-240 Gb/s PAM4 transmitter with 0.70 pJ/b analog power efficiency; good datapoint for the cost of very high per-lane rate.",
  },
  "11409313": {
    priority: "A",
    categories: ["112g", "168g", "pam4", "pam8", "dac_tx"],
    thesis_links: ["lane_rate_tradeoff", "tx_dac_driver_burden"],
    rationale:
      "PAM4/PAM8 DAC-based transmitter with FFE; useful for comparing high-order electrical modulation against low-baud WDM approaches.",
  },
  "11409257": {
    priority: "A",
    categories: ["memory_interface", "pam3", "chiplet_io", "single_ended"],
    thesis_links: ["optical_memory_demo", "memory_bus_extension", "low_swing_bus"],
    rationale:
      "72 Gb/s/pin single-ended PAM3 transceiver for chiplet and memory interfaces; directly maps to memory-bus-extension thinking.",
  },
  "11408977": {
    priority: "A",
    categories: ["memory_interface", "112g", "pam4", "bandwidth_density"],
    thesis_links: ["optical_memory_demo", "package_beachfront"],
    rationale:
      "47 Tb/s/mm and 112 Gb/s/pin memory-interface transceiver; strong benchmark for beachfront and crosstalk constraints.",
  },
  "11409172": {
    priority: "A",
    categories: ["memory_interface", "nrz", "crosstalk_cancellation", "low_energy"],
    thesis_links: ["low_swing_optical_bus", "memory_bus_extension"],
    rationale:
      "100 Gb/s aggregate single-ended NRZ with 1.92 pJ/b and crosstalk cancellation; relevant to NRZ memory-like optical bus assumptions.",
  },
  "11409057": {
    priority: "A",
    categories: ["memory_interface", "nrz", "clocking", "supply_noise"],
    thesis_links: ["timing_spine", "memory_bus_extension", "low_energy_io"],
    rationale:
      "16 Gb/s/pin 0.51 pJ/b NRZ receiver with DQS recovery and clock duty-cycle calibration; close to low-baud bus-extension design space.",
  },
  "11409155": {
    priority: "A",
    categories: ["memory_interface", "low_energy_rx", "isi_robustness"],
    thesis_links: ["low_baud_parallel_io", "memory_bus_extension"],
    rationale:
      "0.092 pJ/b receiver and long-tail ISI robustness; important electrical lower-bound benchmark for low-baud optical alternatives.",
  },
  "11409228": {
    priority: "B",
    categories: ["memory_interface", "parallel_receiver", "multi_drop"],
    thesis_links: ["memory_bus_extension", "low_baud_parallel_io"],
    rationale:
      "Parallel receiver for multi-drop memory interfaces; useful architecture analog for many-lane/memory-semantic optical links.",
  },
  "11409153": {
    priority: "B",
    categories: ["hbm", "memory_interface", "phase_tracking"],
    thesis_links: ["timing_spine", "hbm_adjacent_io"],
    rationale:
      "HBM DQ receiver with baud-rate phase tracking; relevant timing-recovery detail for memory-adjacent I/O.",
  },
  "11409030": {
    priority: "A",
    categories: ["ai_accelerator", "gpu", "3d_stacked", "chiplet"],
    thesis_links: ["ai_system_context", "chiplet_io_demand"],
    rationale:
      "AMD MI350 GPU paper; important demand-side artifact for chiplet/HBM/I/O architecture constraints in AI accelerators.",
  },
  "11409003": {
    priority: "A",
    categories: ["ai_soc", "quad_chiplet", "ucie", "mesh"],
    thesis_links: ["optical_memory_demo", "scale_up_fabric"],
    rationale:
      "Quad-chiplet AI SoC with UCIe-Advanced mesh; strong architecture reference for scale-up electrical fabrics optics would need to complement or displace.",
  },
  "11409090": {
    priority: "B",
    categories: ["ai_accelerator", "enterprise_inference"],
    thesis_links: ["ai_system_context", "inference_memory_pressure"],
    rationale:
      "Inference-optimized accelerator; relevant for workload and memory/communication pressure, less directly an optical I/O paper.",
  },
  "11409201": {
    priority: "B",
    categories: ["reticle_scale", "ai_accelerator"],
    thesis_links: ["ai_system_context", "scale_up_fabric"],
    rationale:
      "Reticle-scale AI accelerator; useful comparator for on-package/on-reticle communication alternatives to optical escape.",
  },
  "11409211": {
    priority: "B",
    categories: ["llm_accelerator", "reram_on_logic", "stacked_memory"],
    thesis_links: ["inference_memory_pressure", "optical_memory_demo"],
    rationale:
      "Stacked LLM accelerator with ReRAM-on-logic; useful as an alternate memory-near-compute approach and workload evidence.",
  },
  "11409184": {
    priority: "A",
    categories: ["near_memory", "3d_dram_logic", "edge_llm"],
    thesis_links: ["optical_memory_demo", "memory_wall"],
    rationale:
      "3D two-DRAM-one-logic process-near-memory chip for edge LLMs; directly relevant to memory-wall alternatives and bandwidth-density framing.",
  },
  "11409139": {
    priority: "A",
    categories: ["hbm4", "dram", "memory_bandwidth"],
    thesis_links: ["memory_wall", "hbm_pressure", "optical_memory_demo"],
    rationale:
      "36 GB HBM4 with 3.3 TB/s bandwidth; core memory-wall benchmark for optical memory/fabric positioning.",
  },
  "11409121": {
    priority: "B",
    categories: ["lpddr6", "dram", "clocking"],
    thesis_links: ["remote_memory_demo", "memory_interface_clocking"],
    rationale:
      "LPDDR6 with WCK tree and 14.4 Gb/s/pin; useful for remote-memory demo baselines and timing/power details.",
  },
  "11409018": {
    priority: "B",
    categories: ["lpddr6", "nrz", "dram"],
    thesis_links: ["remote_memory_demo", "low_baud_parallel_io"],
    rationale:
      "LPDDR6 NRZ signaling and reliability features; relevant to low-baud many-lane memory-interface assumptions.",
  },
  "11409122": {
    priority: "B",
    categories: ["gddr7", "inference_memory", "clocking"],
    thesis_links: ["inference_memory_pressure", "memory_bandwidth"],
    rationale:
      "GDDR7 for mid-range inference AI; useful market/workload evidence for memory bandwidth as an inference bottleneck.",
  },
  "11409328": {
    priority: "B",
    categories: ["pll", "jitter", "fractional_n"],
    thesis_links: ["timing_spine", "clocking_baseline"],
    rationale:
      "Low-jitter DPLL paper; timing baseline for evaluating whether optical timing has system value beyond conventional PLL improvement.",
  },
  "11409271": {
    priority: "B",
    categories: ["pll", "jitter", "fractional_n"],
    thesis_links: ["timing_spine", "clocking_baseline"],
    rationale:
      "21.6 fsrms jitter PLL; relevant state-of-art timing block benchmark.",
  },
  "11408980": {
    priority: "B",
    categories: ["frequency_synthesizer", "jitter", "spur"],
    thesis_links: ["timing_spine", "clocking_baseline"],
    rationale:
      "46.2 fs jitter frequency synthesizer; useful for clock-quality baseline but less directly tied to optical fabric architecture.",
  },
  "11409224": {
    priority: "B",
    categories: ["delay_line", "switched_capacitor", "rf_timing"],
    thesis_links: ["timing_spine", "delay_alignment"],
    rationale:
      "Programmable passive delay line; potentially relevant for timing/phase alignment primitives in burst-mode links.",
  },
  "11409215": {
    priority: "B",
    categories: ["forum", "electrical_links", "optical_links", "400g_plus"],
    thesis_links: ["market_context", "lane_rate_tradeoff"],
    rationale:
      "Forum on electrical and optical links beyond 400G; useful context entry even though it is not a technical paper.",
  },
  "11408957": {
    priority: "B",
    categories: ["short_course", "optical_subsystems"],
    thesis_links: ["team_learning", "optical_subsystem_context"],
    rationale:
      "Short course on optical subsystems; useful background source for a team-building/literature-map JSON, not a primary paper.",
  },
};

const cache = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const byId = new Map(cache.papers.map((paper) => [paper.article_number, paper]));

const selected = Object.entries(selection).map(([articleNumber, meta]) => {
  const paper = byId.get(articleNumber);
  if (!paper) throw new Error(`Missing article ${articleNumber}`);
  return {
    article_number: paper.article_number,
    title: paper.title,
    url: paper.url,
    pdf_url: paper.pdf_url,
    doi: paper.doi,
    doi_url: paper.doi_url,
    publication_title: paper.publication_title,
    publication_date: paper.publication_date,
    pages: {
      start: paper.start_page,
      end: paper.end_page,
    },
    authors: paper.authors,
    priority: meta.priority,
    categories: meta.categories,
    thesis_links: meta.thesis_links,
    rationale: meta.rationale,
    abstract: paper.abstract,
  };
});

const categoryRank = [
  "optical_io",
  "photonic_interposer",
  "coherent_lite",
  "silicon_photonics",
  "ucie",
  "die_to_die",
  "memory_interface",
  "ai_accelerator",
  "near_memory",
  "hbm4",
  "pll",
  "forum",
  "short_course",
];

function rankPaper(paper) {
  const priorityRank = paper.priority === "A" ? 0 : 1;
  const category = paper.categories.find((name) => categoryRank.includes(name));
  const catRank = category ? categoryRank.indexOf(category) : categoryRank.length;
  return [priorityRank, catRank, paper.article_number];
}

selected.sort((a, b) => {
  const ra = rankPaper(a);
  const rb = rankPaper(b);
  return ra[0] - rb[0] || ra[1] - rb[1] || ra[2].localeCompare(rb[2]);
});

const output = {
  generated_at: new Date().toISOString(),
  source_cache: path.relative(process.cwd(), inputPath),
  source_notes_read: [
    "BP/BizPlan-20260607.md",
    "Strategy_Notes/2026-06-02__optical-timing-ocs-system-thesis.md",
    "Strategy_Notes/2026-06-07__optical-remote-memory-demo-plan.md",
    "Strategy_Notes/2026-06-07__wdm-lane-rate-lane-count-tradeoff.md",
    "Strategy_Notes/2026-06-13__stacked-low-baud-dwdm-tfln-pic.md",
    "Literatures/CACHE_INDEX.md",
  ],
  selection_lens: {
    company_thesis:
      "Selection is based on an optical memory, scheduled OCS, and AI communication-fabric thesis enabled by TFLN-on-SiPh, low-swing/direct-drive optical I/O, WDM scaling, timing-aware/burst-mode links, and packaging/system validation.",
    high_priority_rules: [
      "Direct optical I/O, WDM, photonic-interposer, electro-optical routing, or coherent-lite links.",
      "D2D/chiplet/memory-interface circuits that set electrical energy, latency, timing, bandwidth-density, or package-reach baselines.",
      "AI accelerator and memory papers that clarify demand-side bandwidth, memory, chiplet, or scale-up constraints.",
      "Clocking/CDR/PLL/delay papers only when they inform the timing-spine thesis or the SerDes/retiming tax.",
    ],
    exclusions:
      "Excluded most biomedical, RF phased-array, display, sensor, security, and generic CIM papers unless they directly affect optical fabric, memory wall, timing, or AI interconnect positioning.",
  },
  external_browsing_checks: [
    {
      url: "https://submissions.mirasmart.com/ISSCC2026/PDF/ISSCC2026AdvanceProgram.pdf",
      note: "ISSCC 2026 advance program confirms session placement and title for the electro-optical router and related ISSCC sessions.",
    },
    {
      url: "https://www.researchgate.net/publication/401529568_A_32_Gbsl_256GbSFiber_Half-Rate_Bandpass-Filtered_Clock-Forwarding_DWDM_Optical_Link_in_a_3D-Stacked_7nm_EIC65nm_PIC_Technology",
      note: "Public indexed listing confirms DOI/title/authors for the 32 Gb/s/lambda clock-forwarding DWDM 3D-stacked EIC/PIC paper.",
    },
    {
      url: "https://www.researchgate.net/publication/401528678_A_319pJb_Electro-Optical_Router_with_18ns_Setup_Frame-Level_Routing_and_1-to-6_Wavelength-Flexible_Link_Capacity_for_Photonic_Interposers",
      note: "Public indexed listing confirms DOI/title/authors for the 3.19 pJ/b electro-optical router paper.",
    },
    {
      url: "https://www.inelectronics.co.uk/cea-shows-18-ns-electro-optical-router-for-chiplets/",
      note: "External technical news article summarizes the CEA electro-optical router prototype metrics: 18 ns setup, 3.19 pJ/b, photonic interposer context.",
    },
    {
      url: "https://www.researchgate.net/publication/401533072_232_A_2-Channel_800Gbs_Transceiver_for_Coherent-Lite_Applications_with",
      note: "Public indexed listing confirms DOI/title/authors for the 2-channel 800 Gb/s coherent-lite transceiver paper.",
    },
    {
      url: "https://www.linkedin.com/posts/enrico-monaco-524b486_presented-at-the-international-solid-state-circuits-activity-7429886000962658304-mcZY",
      note: "Author-side public post describes the coherent-lite transceiver as a 2-20 km data-center-interconnect architecture with O-band optics and sub-300 ns latency.",
    },
  ],
  counts: {
    selected_total: selected.length,
    priority_a: selected.filter((paper) => paper.priority === "A").length,
    priority_b: selected.filter((paper) => paper.priority === "B").length,
  },
  selected_papers: selected,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`);
console.log(`Wrote ${selected.length} selected papers to ${outputPath}`);
