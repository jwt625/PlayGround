#!/usr/bin/env python3
"""Generate manifest entries for OFS batch papers."""
import json
from datetime import datetime

papers = [
    {"file": "anand_2024_microwave_photon_detection.pdf", "title": "Microwave single-photon detection using a hybrid spin-optomechanical quantum interface", "year": 2024, "url": "https://arxiv.org/abs/2401.10455"},
    {"file": "bao_2024_cryogenic_pulse_generator.pdf", "title": "A cryogenic on-chip microwave pulse generator for large-scale superconducting quantum computing", "year": 2024, "url": "https://arxiv.org/abs/2402.18645"},
    {"file": "bland_2025_2d_transmon_1ms.pdf", "title": "2D transmons with lifetimes and coherence times exceeding 1 millisecond", "year": 2025, "url": "https://arxiv.org/abs/2503.14798"},
    {"file": "bolgar_2025_airbridge_stiffeners.pdf", "title": "Highly stable aluminum air-bridges with stiffeners", "year": 2025, "url": "https://arxiv.org/abs/2504.07810"},
    {"file": "bosonic_cqed_review_2023.pdf", "title": "Shaping photons: quantum information processing with bosonic cQED", "year": 2023, "url": "https://arxiv.org/abs/2311.03846"},
    {"file": "brell_2025_lowdepth_unitary.pdf", "title": "Low-Depth Unitary Quantum Circuits for Dualities in One-Dimensional Quantum Lattice Models", "year": 2025, "url": "https://arxiv.org/abs/2411.03562"},
    {"file": "bruckmoser_2025_nb_airbridges.pdf", "title": "Niobium Air Bridges as a Low-Loss Component for Superconducting Quantum Hardware", "year": 2025, "url": "https://arxiv.org/abs/2503.12076"},
    {"file": "brunner_2025_neuromorphic_photonics_roadmap.pdf", "title": "Roadmap on Neuromorphic Photonics", "year": 2025, "url": "https://arxiv.org/abs/2501.07917"},
    {"file": "chang_2025_jtwpa.pdf", "title": "Josephson traveling-wave parametric amplifier based on low-intrinsic-loss coplanar lumped-element waveguide", "year": 2025, "url": "https://arxiv.org/abs/2503.07559"},
    {"file": "chen_2014_aluminum_airbridges.pdf", "title": "Fabrication and characterization of aluminum airbridges for superconducting microwave circuits", "year": 2014, "url": "https://arxiv.org/abs/1310.2325"},
    {"file": "chou_2024_multiphonon_entanglement.pdf", "title": "Deterministic multi-phonon entanglement between two mechanical resonators on separate substrates", "year": 2024, "url": "https://arxiv.org/abs/2411.15726"},
    {"file": "corcoles_2011_protecting_qubits.pdf", "title": "Protecting superconducting qubits from external sources of loss and heat", "year": 2011, "url": "https://arxiv.org/abs/1108.1383"},
    {"file": "croot_2025_scalable_sc_qc.pdf", "title": "Enabling Technologies for Scalable Superconducting Quantum Computing", "year": 2025, "url": "https://arxiv.org/abs/2512.15001"},
    {"file": "dhundhwal_2025_tantalum_resonators.pdf", "title": "High quality superconducting tantalum resonators with beta phase defects", "year": 2025, "url": "https://arxiv.org/abs/2502.17247"},
    {"file": "diego_2026_lamb_wave.pdf", "title": "Gigahertz-frequency Lamb wave resonator cavities on suspended lithium niobate for quantum acoustics", "year": 2026, "url": "https://arxiv.org/abs/2601.13509"},
    {"file": "fu_2026_airbridge_single_step.pdf", "title": "Low-Loss, High-Coherence Airbridge Interconnects Fabricated by Single-Step Lithography", "year": 2026, "url": "https://arxiv.org/abs/2601.16416"},
    {"file": "gargiulo_2021_magnetic_hose.pdf", "title": "Fast flux control of 3D transmon qubits using a magnetic hose", "year": 2021, "url": "https://arxiv.org/abs/2010.02668"},
    {"file": "google_2024_qec_below_surface_code_threshold.pdf", "title": "Quantum error correction below the surface code threshold", "year": 2024, "url": "https://arxiv.org/abs/2408.13687"},
    {"file": "groh_2025_tes_multiplexer.pdf", "title": "Demonstration of a 1820 channel multiplexer for transition-edge sensor bolometers", "year": 2025, "url": "https://arxiv.org/abs/2507.10929"},
    {"file": "hatinen_2023_flipchip_thermal.pdf", "title": "Thermal resistance in superconducting flip-chip assemblies", "year": 2023, "url": "https://arxiv.org/abs/2303.01228"},
    {"file": "howe_2022_jpg_qubit_control.pdf", "title": "Digital Control of a Superconducting Qubit Using a Josephson Pulse Generator at 3 K", "year": 2022, "url": "https://arxiv.org/abs/2109.04769"},
    {"file": "huang_2025_dualrail_entanglement.pdf", "title": "Logical multi-qubit entanglement with dual-rail superconducting qubits", "year": 2025, "url": "https://arxiv.org/abs/2504.12099"},
    {"file": "ihssen_2025_flipchip_low_crosstalk.pdf", "title": "Low crosstalk modular flip-chip architecture for coupled superconducting qubits", "year": 2025, "url": "https://arxiv.org/abs/2502.19927"},
    {"file": "leonard_2019_digital_coherent_control.pdf", "title": "Digital Coherent Control of a Superconducting Qubit", "year": 2019, "url": "https://arxiv.org/abs/1806.07930"},
    {"file": "li_2025_superfluid_transmon.pdf", "title": "Superfluid-Cooled Transmon Qubits under Optical Excitation", "year": 2025, "url": "https://arxiv.org/abs/2501.08375"},
    {"file": "liu_2025_threequbit_gates.pdf", "title": "Direct Implementation of High-Fidelity Three-Qubit Gates for Superconducting Processor with Tunable Couplers", "year": 2025, "url": "https://arxiv.org/abs/2506.16124"},
    {"file": "magnard_2020_remote_entanglement.pdf", "title": "Deterministic remote entanglement of superconducting circuits through microwave two-photon transitions", "year": 2020, "url": "https://arxiv.org/abs/2005.00773"},
    {"file": "magnard_2021_state_transfer.pdf", "title": "Deterministic quantum state transfer between distant superconducting qubits", "year": 2021, "url": "https://arxiv.org/abs/2004.01425"},
    {"file": "malevannaya_2025_shielding_guide.pdf", "title": "An engineering guide to superconducting quantum circuit shielding", "year": 2025, "url": "https://arxiv.org/abs/2504.08700"},
    {"file": "mcdermott_2014_sfq_qubit_control.pdf", "title": "Accurate Qubit Control with Single Flux Quantum Pulses", "year": 2014, "url": "https://arxiv.org/abs/1402.4029"},
    {"file": "meng_2022_fractal_snspd.pdf", "title": "Fractal Superconducting Nanowires Detect Infrared Single Photons with 84% System Detection Efficiency", "year": 2022, "url": "https://arxiv.org/abs/2102.07811"},
    {"file": "mohl_2025_microwave_optical_transducer.pdf", "title": "Bidirectional microwave-optical conversion using an integrated barium-titanate transducer", "year": 2025, "url": "https://arxiv.org/abs/2501.09728"},
    {"file": "mohseni_2024_quantum_supercomputer_scaling.pdf", "title": "How to Build a Quantum Supercomputer: Scaling Challenges and Opportunities", "year": 2024, "url": "https://arxiv.org/abs/2411.10406"},
    {"file": "nagasawa_2014_nb_9layer.pdf", "title": "Nb 9-Layer Fabrication Process for Superconducting Large-Scale SFQ Circuits and Its Process Evaluation", "year": 2014, "url": "https://doi.org/10.1587/transele.E97.C.132"},
    {"file": "nakamura_1999_charge_qubit.pdf", "title": "Coherent control of macroscopic quantum states in a single-Cooper-pair box", "year": 1999, "url": "https://arxiv.org/abs/cond-mat/9904003"},
    {"file": "piezoelectricity_2024.pdf", "title": "Observation of Interface Piezoelectricity in Superconducting Devices on Silicon", "year": 2024, "url": "https://arxiv.org/abs/2409.10626"},
    {"file": "prx_2021_josephson_entangled_beams.pdf", "title": "Generating Two Continuous Entangled Microwave Beams Using a dc-Biased Josephson Junction", "year": 2021, "url": "https://arxiv.org/abs/2012.01203"},
    {"file": "putterman_2025_bosonic_qec.pdf", "title": "Hardware-efficient quantum error correction via concatenated bosonic qubits", "year": 2025, "url": "https://arxiv.org/abs/2409.13025"},
    {"file": "qiu_2023_quantum_teleportation_arxiv.pdf", "title": "Deterministic quantum state and gate teleportation between distant superconducting chips", "year": 2023, "url": "https://arxiv.org/abs/2302.08756"},
    {"file": "ranadive_2025_twpa_isolator.pdf", "title": "A travelling-wave parametric amplifier isolator", "year": 2025, "url": "https://arxiv.org/abs/2401.15398"},
    {"file": "roth_2022_transmon_tutorial.pdf", "title": "The Transmon Qubit for Electromagnetics Engineers: An introduction", "year": 2022, "url": "https://arxiv.org/abs/2208.05114"},
    {"file": "rousseau_2025_cat_qubit_squeezing.pdf", "title": "Enhancing dissipative cat qubit protection by squeezing", "year": 2025, "url": "https://arxiv.org/abs/2502.07892"},
    {"file": "russo_2025_sc_esr_detector.pdf", "title": "Superconducting microwave oscillators as detectors for ESR spectroscopy", "year": 2025, "url": "https://arxiv.org/abs/2503.16587"},
    {"file": "shen_2024_sfq_photonic_link.pdf", "title": "Photonic link from single-flux-quantum circuits to room temperature", "year": 2024, "url": "https://arxiv.org/abs/2308.01302"},
    {"file": "smith_2024_single_qubit_gates_10e7.pdf", "title": "Single-qubit gates with errors at the 10^-7 level", "year": 2024, "url": "https://arxiv.org/abs/2412.04421"},
    {"file": "song_2026_thermal_annealing.pdf", "title": "Improving coherence and stability of superconducting qubits by thermal annealing in vacuum", "year": 2026, "url": "https://arxiv.org/abs/2601.15580"},
    {"file": "storz_2023_teleportation_30m.pdf", "title": "Deterministic quantum state and gate teleportation between distant superconducting chips", "year": 2023, "url": "https://arxiv.org/abs/2212.01637"},
    {"file": "sun_2025_floquet_bacon_shor.pdf", "title": "Logical Operations with a Dynamical Qubit in Floquet-Bacon-Shor Code", "year": 2025, "url": "https://arxiv.org/abs/2511.05382"},
    {"file": "tian_2025_500qubit_wiring.pdf", "title": "High-density wiring solution for 500-qubit scale superconducting quantum processors", "year": 2025, "url": "https://arxiv.org/abs/2506.04920"},
    {"file": "vedral_2021_tardigrade_qubit_entanglement.pdf", "title": "Entanglement between superconducting qubits and a tardigrade", "year": 2021, "url": "https://arxiv.org/abs/2112.07978"},
    {"file": "yamanashi_2010_sfq_100ghz.pdf", "title": "100 GHz Demonstrations Based on the Single-Flux-Quantum Cell Library for the 10 kA/cm2 Nb Multi-Layer Process", "year": 2010, "url": "https://ynu.repo.nii.ac.jp/?action=repository_uri&item_id=8802"},
    {"file": "yan_2025_quantum_secret_sharing.pdf", "title": "Quantum secret sharing in a triangular superconducting quantum network", "year": 2025, "url": "https://arxiv.org/abs/2506.10878"},
    {"file": "youssefi_2025_dirac_cone_sc_circuits.pdf", "title": "Realization of tilted Dirac-like microwave cone in superconducting circuit lattices", "year": 2025, "url": "https://arxiv.org/abs/2501.10434"},
    {"file": "youssefi_2025_vacuum_gap_capacitors.pdf", "title": "Compact superconducting vacuum-gap capacitors with low microwave loss and high mechanical coherence for scalable quantum circuits", "year": 2025, "url": "https://arxiv.org/abs/2501.03211"},
    {"file": "zhou_2025_km_photonic_link.pdf", "title": "A kilometer photonic link connecting superconducting circuits in two dilution refrigerators", "year": 2025, "url": "https://arxiv.org/abs/2508.02444"},
]

base_path = "semiconductor_processing_dataset/raw_documents/papers/superconducting_qubits/ofs_batch"
timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

for p in papers:
    doc_id = p["file"].replace(".pdf", "")
    entry = {
        "document_id": doc_id,
        "source_type": "journal_article",
        "title": p["title"],
        "institution": None,
        "year": p["year"],
        "url": p["url"],
        "source_path": f"{base_path}/{p['file']}",
        "status": "succeeded",
        "last_attempt_at": timestamp,
        "attempt_count": 1,
        "last_error": None,
        "quality_assessment": "high_value",
        "priority": "medium",
        "notes": "OFS batch extraction - Phase 1B"
    }
    print(json.dumps(entry))

