# OFC 2026 High-Speed, Heterogeneous-Integration, and 1.6T+ Papers

Source: local cached OFC metadata in `output/full_metadata/ofc_full_metadata.csv` and extracted paper text in `extracted_text/`.

## High Baud / Lane-Rate Demos, Including SiPh, LN, LT, BTO

| Code | Presentation / paper | Affiliation(s) | Why it matches |
|---|---|---|---|
| `Th4B.3` | Barium Titanate Enabling Net 1.6T (4x448 Gbps PAM4) on a Silicon Photonics Platform | McGill University; Ciena Corporation; Lumiphase AG | BTO DR4 chip on commercial SiPh; net 1.6T, 4x448G PAM4 |
| `Th4B.2` | Driver-Less 448 Gbps PAM4 and 1.2 Tbps 16-QAM IMDD/Coherent-Lite Transmission Using TFLN Optical DACs | McGill University; HyperLight Corporation; Ciena Corporation | TFLN optical DAC; 448G PAM4 IM/DD, 1.2T 16QAM |
| `W4J.4` | A 420 Gb/s/Lane O-Band PAM-4 TOSA Based on Thin-Film Lithium Niobate for IM-DD Applications | Wuhan HG Genuine Optics Tech Co., Ltd. | TFLN TOSA; 210 GBaud, 420G/lane PAM4 |
| `Th4A.2` | C-Band 110-GHz-Bandwidth Thin-Film Lithium Tantalate Modulator Enabling 768 (536) Gbit/s Line (Net) Data Rates | University of Michigan; Nokia Bell Labs | LT/TFLT modulator; >110 GHz EO BW; 768G line / 536G net |
| `Th1C.3` | 400G/Lane for Linear-Drive Optics Applications | Ligent Technologies Inc | TFLN MZM; 400G/lane for LPO/NPO/CPO |
| `Th4A.4` | 400G/Lane PAM4 Modulation Using Silicon Mach-Zehnder Modulators | Coherent Corp | Silicon MZM plus commercial SiGe driver; 400G/lane PAM4 |
| `Th4A.1` | 4-ch x 400-Gbps PAM4 O-Band Membrane InGaAlAs EA-DFB Laser Array on a Si Photonics Platform | Device Innovation Center, NTT, Inc. | InGaAlAs EA-DFB on SiPh; 400/448G PAM4 per channel |
| `W1A.7` | A 1.6 Tbit/s WDM Integrated Photonic IMDD Transmitter on Thin-Film Lithium Tantalate | Swiss Federal Institute of Technology; Karlsruhe Institute of Technology | LT transmitter; aggregate 1.6T WDM IM/DD |
| `M2B.4` | Ferroelectric Nematic Glass-Based Silicon Photonics Modulator for Net 400 Gbps IM/DD Transmission | McGill University; Polaris Electro-Optics Inc | Hybrid Si + ferroelectric nematic glass; 200 GBaud PAM6 / 168 GBaud PAM8 |
| `M2B.5` | O-Band Silicon-Plasmonic Resonant Ring Modulator Demonstrating Net-Rates of 400 Gbps | ETH Zurich, Institute of Electromagnetic Fields | SiPh-plasmonic ring; net 400G with PAM8 |
| `Th3J.4` | Barium Titanate DP-IQM Enabling Net 1 Tbps/lambda ZR and Coherent-Lite Data Center Networks | McGill University | BTO coherent modulator; net 1T/lambda |
| `Th3J.5` | Silicon-Organic Hybrid IQ Modulators With Sub-1 V pi-Voltages Operating at 200 GBd 16QAM and 144 GBd 64QAM | Karlsruhe Institute of Technology (KIT) | SOH/SiPh coherent IQ; 200 GBd 16QAM |
| `W3E.6` | 180 GBaud PAM4 Driver-Modulator Engine for IM/DD Transmissions in the O-Band | Fraunhofer HHI | Co-packaged InP MZM + driver; 180 GBaud PAM4 |
| `W1A.3` | Thin-Film Lithium Niobate Modulators With 110 GHz Bandwidth and 1.9 V-cm Efficiency on 200-mm Silicon Substrate | National Semiconductor Translation and Innovation Centre | TFLN on 200-mm Si; 110 GHz BW |
| `W1A.4` | Sub-V-Driven 110-GHz O-Band Electro-Optic Modulator on Thin-Film Lithium Tantalate | University of Michigan | LT O-band MZM; sub-1V, >110 GHz |
| `Th4A.6` | TFLN-Based Wafer-Level Co-Packaged Optics Engine for Ultrahigh-Bandwidth Electro-Optical Modulation | Shanghai Jiao Tong University | TFLN CPO engine; >100 GHz bandwidth |

## Heterogeneous / Hybrid Integration of LN, LT, InP, SiPh

| Code | Presentation / paper | Affiliation(s) | Integration angle |
|---|---|---|---|
| `Th1D.1` | Photonics Heterogeneous Integration of TFLN and Hydrogen-Free SiN on a 200-mm Silicon Photonics Platform | Institute of Microelectronics (IME); National Semiconductor Translation and Innovation Centre | Die-to-wafer TFLN + SiN on 200-mm SiPh |
| `M4D.2` | IQ Modulators With Two Segments Using InP/SOI Chip-on-Wafer Bonding Process for Optical DAC | Photonics Electronics Technology Research Association; Sumitomo Electric | InP/Si chip-on-wafer bonding |
| `M4D.4` | Truly-Differential Drive of TFLN TWE-MZM by Linear SiGe Driver in a Codesigned Hybrid Integrated Assembly | Nokia Bell Labs | Hybrid TFLN modulator + SiGe driver |
| `Th2A.1` | Hybrid Integration of O-Band InP SOA Array and PLC Using PLC/SiN Spot-Size Converter | Eindhoven University of Technology | InP SOA array + PLC/SiN coupler |
| `Th4A.1` | 4-ch x 400-Gbps PAM4 O-Band Membrane InGaAlAs EA-DFB Laser Array on a Si Photonics Platform | Device Innovation Center, NTT, Inc. | Membrane III-V/InGaAlAs laser array on SiPh |
| `Th4B.3` | BTO Enabling Net 1.6T on a Silicon Photonics Platform | McGill University; Ciena Corporation; Lumiphase AG | Thin-film BTO monolithically integrated on SiPh |
| `M4B.2` | A 256 Gb/s DWDM Optical I/O in a 3D-Stacked EIC/PIC Silicon Photonics Platform | NVIDIA Corporation | 3D-stacked 7 nm EIC / 65 nm SiPh PIC |
| `M4D.6` | Wafer-Scale Transfer Printing: Ready for Production, Ready for Impact | Universiteit Gent | General heterogeneous transfer printing platform |
| `Th3C.2` | Integrated Glass Waveguide Substrate With Surface Coupled Photonic Chips for Massive Scaling of CPO | Corning Optical Communication GmbH & Co. KG; Corning Technology Center Korea | Photonic-chip coupling/interposer approach for CPO scaling |
| `Th1G.4` | Multicore Fiber Coupled Backside-Emitting VCSEL/PD Arrays for High-Bandwidth Optical Interconnects | LightXcelerate Inc | VCSEL/PD arrays flip-bonded to driver/TIA EICs |

## 1.6T and Beyond / System-Level High-Speed Demos

| Code | Presentation / paper | Affiliation(s) | Scale demonstrated |
|---|---|---|---|
| `Th4A.7` | 1.6T (8x200Gb/s) 2xFR4 Silicon Photonic IMDD Transceiver | HGGenuine Optics Tech Co., Ltd. | Monolithic 1.6T SiPh transceiver in OSFP module |
| `Th4B.3` | BTO Enabling Net 1.6T (4x448 Gbps PAM4) on SiPh | McGill University; Ciena Corporation; Lumiphase AG | Net 1.6T with 4x448G PAM4 |
| `Th1C.4` | 1.6 Tb/s Monolithic InP Transmitter PIC With DFB, MZM, and SOA Arrays | Nokia Corporation | 8-channel monolithic InP PIC; 1.6T class |
| `M4B.5` | Highly-Integrated 16-Channel Silicon-Photonics Optical Engine Enabling PAM6 Transmission | Ciena Corporation | >2 Tb/s aggregate SiPh optical engine |
| `W3J.1` | Real-Time S+C+L-Band 134-Tb/s DCI Bidi Transmission With All Channels at 1.2-Tb/s Enabled by SiPh Transceiver | ZTE Corporation; China Mobile Research Institute | 134T DCI; 1.276T/channel |
| `W1C.2` | High-Symbol-Rate 192-GBaud Signal Transmission in S+C+L Band Over 2000 km With Net Bitrate of 102.8 Tb/s | NTT, Inc. | 192 GBaud, 102.8T over 2000 km |
| `M2C.5` | Real-Time C+L Band Transmission of 64 Tb/s With 1 Tb/s Single-Carrier Channels Over 1260 km | Cisco Systems Inc.; Corning Incorporated | 64x1T real-time channels |
| `M2C.3` | Real-Time Unrepeatered Transmission of 400G/800G/1.2T Over HCF | China Telecom Research Institute; YOFC | 1.2T unrepeatered over 436.1 km HCF |
| `Th2A.46` | Net 5.8 Tbps IM/DD Transmission Over 2 km Using 25 Comb Channels and a Single SOA | McGill University | 3.2T PAM4, 4.3T PAM6, 5.76T PAM8 |
| `Th3A.2` | Real-Time 2.5-Pb/s Bidirectional Transmission Over 24-Core Single-Mode Fiber | China Information Communication Technologies Group Corporation; Peng Cheng Laboratory | 2.5 Pb/s real-time SDM/WDM |
| `Th1J.6` | Beyond 550 Tb/s S+C+L-Band Bidirectional Transmission Over 10.9-km AR-HCF | Pengcheng Laboratory; Central China Normal University | 550.97T net estimated |
| `Th4B.5` | 450 Tb/s GMI, 42.4 THz, OESCL-Band Transmission Over Field-Deployed Fiber | NICT; University College London | >450T GMI, 418T decoded |
| `Th4B.7` | Sparsely Repeated 21.7 Tb/s Net-Rate Transoceanic Transmission With HCF | Nokia Bell Labs France | 21.7T over 6660 km |
| `M1H.1` | Over-3-Tb/s/lambda Free-Space MIMO Transmission Under Diffraction | NTT, Inc. | 3.6T/lambda FSO |
| `M3F.3` | 4,096x4,096 Optical Circuit Switch Delivering 819.2 Tb/s | Nagoya University; Toyota Technological Institute | AI-networking OCS demo, 819.2T switching layer |

## Optical Circuit Switching (OCS) / Optical AI Fabrics

| Code | Presentation / paper | Affiliation(s) | OCS angle |
|---|---|---|---|
| `M2D.3` | High Port Count Silicon Photonic MEMS Circuit Switch | nEye.ai | SiPh MEMS OCS device requirements: switching speed, power, polarization, and loss |
| `M3F.2` | Experimental Demonstration of O-Band 4x4x8lambda Wavelength Selective Switch at 100Gbps/lambda for Data Center Networks | Technische Universiteit Eindhoven; Huawei Technologies Duesseldorf GmbH | O-band modular WSS for data-center optical switching |
| `M3F.3` | A 4,096x4,096 Strictly Non-Blocking Optical Circuit Switch Delivering 819.2 Tb/s via Space-and-Wavelength Routing | Nagoya University; Toyota Technological Institute | Large-radix non-blocking OCS using space-and-wavelength routing |
| `M3F.4` | Reconfiguration-Aware Direct-Connect AI Cluster Using Spatial-and-Wavelength-Selective Switching | Columbia University; Massachusetts Institute of Technology | AI cluster integrated with spatial/wavelength switching and Linux network stack |
| `M3F.5` | Training-Phase-Aware Optical Circuit Switching Reconfiguration for Large Language Model | University of Cambridge; imec | Phase-aware OCS reconfiguration for LLM training collectives |
| `M3F.6` | Optical Switching for AI Factories | NVIDIA Corporation | System/architecture perspective on optical switching for large GPU clusters |
| `M4F.2` | Accelerating LLM Training in Optical AI Clusters With Asynchronously-Invoked Hitless in-Job Partial TPE | University of Science and Technology of China | Hitless partial OCS topology reconfiguration during LLM training |
| `W2A.28` | Performance Thresholds for Optical Circuit Switching in LLM Inference | University of California Berkeley | OCS reconfiguration-speed thresholds for LLM inference networks |
| `W2A.31` | OCS-Based Double Resource Pooling for Flexible Intra- and Inter-Rail Connectivity in AIDC Networks | Shanghai Jiao Tong University | Flat OCS-based AIDC network with double resource pooling and lower power |
| `W2A.41` | PCIe-Over-Optics With OSFP DR8 LPO and an Optical Circuit Switching Fabric for Composable CPU-GPU Resource Pooling | Alibaba Cloud Computing; Ruijie Networks | Dynamic OCS fabric for composable CPU-GPU resource pooling |
| `W4H.3` | Cluster- and Reach-Scalable Optical Switching for Scale-Across AI System | KDDI Research, Inc.; GeNopsys Technologies, Inc. | OCS-based scale-across GPU cluster architecture over metro reach |
| `W4H.4` | Auto-Allocating OCS Based on Real Time Flow-Granularity Controller for LM Training | Beijing University of Posts and Telecommunications; China Unicom Research Institute; Infrawaves; Institute of Computing Technology, CAS | Per-flow OCS controller for LM training hot-flow routing |
| `W4H.6` | Research and Practice of Optical Network Technology for High Reliability Interconnection of Large Scale Data Centers | China Telecom Research Institute; China Telecommunications Corporation | Field/lab validation combining coherent transmission, OCS, and fast rerouting |
| `Th2A.35` | HOCSS: a Hardware-Accelerated Optical Circuit Switch Scheduler for low-Latency Optimal Ports Matching | Institute of Computing Technology, CAS; University of Chinese Academy of Sciences | Hardware-accelerated OCS scheduler with sub-microsecond matching latency |
| `Tu2C.6` | Demonstration of a Collision Control Mechanism for Inter-AIDC Traffic in a Spine-Leaf Multi-Granularity All-Optical Switching Network | Beijing University of Posts & Telecom | FPGA-tested collision control for multi-granularity all-optical switching |

## Scale-Up: AI Pods, Architecture, and Standards Presentations

These are the most direct "scale-up" schedule matches from the cached OFC metadata. Several are talks/symposium presentations rather than technical-paper PDFs, so the browser may show metadata only.

| Code | Presentation / paper | Affiliation(s) | Scale-up angle |
|---|---|---|---|
| `Tu1A.2` | Revolutionizing Networking for Gigawatt AI Factories | Plenary Session | Million-GPU-scale training framing across compute, memory scale-up, networking scale-up, and scale-out |
| `M4F.6` | Scale-Out and Scale-Up Photonic Interconnects | Photonic Interconnects for Scalable AI | Direct architecture-level talk comparing photonic scale-out and scale-up interconnect roles |
| `M4B.6` | Technology for AI Interconnect Scale-Up Solutions | Optical Engines | Evaluates optical scale-up networking technologies, reliability, and cluster-performance implications |
| `M4B.1` | Integrated Optical I/O Chiplets for Bandwidth Scaling in AI Infrastructure | Optical Engines | Review of SiPh optical I/O chiplets for high-density, power-efficient AI bandwidth scaling |
| `Tu3I.1` | Next Generation Interconnects for AI Scale Up - End User Perspective | Symposium: Next Generation Interconnects for AI Scale-up Systems | End-user view on optical scale-up deployment, larger domains, rack disaggregation, and bandwidth |
| `Tu3I.2` | Practical Challenges in Scaling to Multi-Rack Pods | Symposium: Next Generation Interconnects for AI Scale-up Systems | System constraints for multi-rack scale-up domains: link performance, RAS, physical integration, and fiber management |
| `Tu3I.3` | Next Generation Interconnects for AI Scale Up Systems Symposium Speaker | Symposium: Next Generation Interconnects for AI Scale-up Systems | Placeholder/metadata-only symposium talk; keep as a schedule hit until reviewed |
| `Tu3I.4` | Sensing VCSEL Interconnects for Scaleup Applications | Symposium: Next Generation Interconnects for AI Scale-up Systems | VCSEL-based scale-up interconnects emphasizing cost, reliability, density, and supply chain |
| `Tu3I.5` | Enabling Beyond CPO - Challenges and Opportunities | Symposium: Next Generation Interconnects for AI Scale-up Systems | CPO architecture path toward deeper data-center integration |
| `Tu3I.6` | Advanced Interconnects for Silicon Photonics Packaging | Symposium: Next Generation Interconnects for AI Scale-up Systems | Detachable optical links, 2.5D assembly/test, and SiPh packaging for switch/compute applications |
| `Tu3I.7` | Innovative Electrical and Optical Interconnects in Advanced Packaging | Symposium: Next Generation Interconnects for AI Scale-up Systems | Advanced-package electrical/optical interconnect tradeoffs for AI scale-up systems |
| `Tu3I.8` | Manufacturing and Packaging of Optical Interconnects for AI Scale-up Systems | Symposium: Next Generation Interconnects for AI Scale-up Systems | Manufacturing and packaging process view for optical scale-up interconnects |
| `Th1C.1` | 400G/Lane Signaling Ecosystem and Standard | Data Center Subsystems | Standards/ecosystem talk explicitly driven by AI scale-up and scale-out bandwidth demand |
| `Th1A.1` | Intra-Data Center THz Interconnects | Short Reach Optical and THz Interconnects | Non-optical/THz short-reach option for scale-up domains; 224/448G PAM4 and UCIe-style formats |

## Scale-Up: VCSELs, Micro-LEDs, PCSELs, and Surface-Emitting Sources

| Code | Presentation / paper | Affiliation(s) | Scale-up angle |
|---|---|---|---|
| `Tu3I.4` | Sensing VCSEL Interconnects for Scaleup Applications | Symposium: Next Generation Interconnects for AI Scale-up Systems | Direct VCSEL scale-up talk; leverages sensing-VCSEL supply chain and dense low-power links |
| `Th1G.4` | Multicore Fiber Coupled Backside-Emitting VCSEL/PD Arrays for High-Bandwidth Optical Interconnects in Data Center | LightXcelerate Inc | 19-core MCF-coupled optical chiplet with backside VCSEL/PD arrays flip-bonded to driver/TIA EICs |
| `W1B.4` | A 53-Gbaud NRZ/PAM4 x 8-Channel 1060-nm Single-Mode VCSEL-Based Ultra-Compact and High-Energy-Efficient CPO Transceiver for Full-Reach Datacenter Links | Furukawa Denki Kogyo Kabushiki Kaisha; Fuji Film Business Innovation Kabushiki Kaisha; Institute of Science Tokyo | 8-channel 1060-nm SM-VCSEL CPO transceiver, 2-km parallel SMF, microlens-array coupling |
| `W1B.5` | 106-Gbps 940-nm Flip-Chip Back-Emitting VCSEL With Metalens for NPO/CPO Applications | Laser Prototypes and Packaging | Back-emitting VCSEL with integrated metalens, high-temperature 106G PAM4, relaxed fiber-coupling tolerance |
| `W1B.6` | Mass Production of VCSELs for Datacom and Sensing Applications | Laser Prototypes and Packaging | Manufacturing/process-control perspective on VCSEL foundry production and reliability |
| `W4E.3` | 200-Gb/s 1060-nm Single-Mode Coupled-Cavity VCSEL Enabling Modal-Dispersion Free >50-GHz Bandwidth Over 500-m Multimode Fiber | Institute of Science Tokyo | High-speed single-mode VCSEL extending MMF reach to 500 m at 200G |
| `W4E.4` | High-Performance, Cost-Effective SWIR VCSELs: a New Source for Optical Interconnects | Advanced Semiconductor Laser Sources | Mass-producible InP SWIR VCSELs for low-power/high-speed optical interconnects |
| `Th4A.3` | High Temperature >35GHz Bandwidth Oxide-Confined VCSELs for 200G-PAM4 Links | Postdeadline Session I | 200G-PAM4 VCSEL reliability/performance at 25-80 C with >10-year lifetime claim |
| `Th1A.2` | 200G/Lane 50-m Multimode VCSEL Link by Low-Material-Dispersion Graded-Index Plastic Optical Fiber | Short Reach Optical and THz Interconnects | Multimode VCSEL link route using low-material-dispersion GI-POF for short-reach scale-up |
| `Th1A.3` | 180 Gb/s PAM-4 Optical Link by Cryogenic VCSEL | Short Reach Optical and THz Interconnects | Cryogenic VCSEL option for low-energy optical links near cold electronics/quantum systems |
| `Th1A.4` | Record 91-mW Cryogenic VCSEL Enabling 140-Gb/s PAM-4 Transmission From 3 K to 40 K for Quantum Optical Interconnects | Short Reach Optical and THz Interconnects | High-power cryogenic VCSEL for 140G PAM4 over 3-40 K |
| `M3H.2` | Demonstrating 80 Gb/s Optical Wireless Communication Using a Multi-Aperture VCSEL and a Multi-Mode Fiber-Coupled Receiver for Next-Generation LiFi Connectivity | Visible Light and Optical Wireless Communication | Multi-aperture VCSEL optical-wireless link; adjacent free-space short-reach approach |
| `Th1G.3` | Low-Power and Short-Distance Wireless Optical Interconnection System Based on 1.6 GHz Bandwidth Red Micro-LED | Next-Generation Optical I/O | Micro-LED/uLED wireless optical interconnect: 1.5 Gbps at 0.22 pJ/bit DC power |
| `W4E.2` | 500 mW O-Band Photonic-Crystal Surface-Emitting Lasers and Scalable 2D Arrays for Multi-Channel CPO Applications | Advanced Semiconductor Laser Sources | PCSEL route: high-power O-band source and 2x2 array for multi-channel CPO |
| `W4E.5` | O-Band Membrane Surface-Emitting Laser on a Si Substrate Demonstrating 100-Gbps PAM-4 Operation | Device Technology Labs., NTT, Inc.; Institute of Integrated Research, Institute of Science Tokyo | Membrane surface-emitting laser on Si substrate; 100G PAM4 over 2-km SSMF |
| `Th1G.5` | Ethernet-Over-OWC Using VCSELs: Transparent Gigabit Links With Low Latency and Robust Alignment Tolerance | Next-Generation Optical I/O | Commercial-component VCSEL-PIN optical-wireless link; not high aggregate rate, but relevant to alignment-tolerant short links |

## Scale-Up: Wafer-Scale and Heterogeneous Manufacturing

| Code | Presentation / paper | Affiliation(s) | Scale-up angle |
|---|---|---|---|
| `M4D.6` | Wafer-Scale Transfer Printing: Ready for Production, Ready for Impact | Universiteit Gent | Production-readiness of wafer-scale micro-transfer printing for III-V and LN on silicon |
| `M4D.5` | 82-mm-Long Optical Link Using Micro-Transfer-Printed Directly Modulated Membrane Laser and Photodetector on SiN Waveguide: Toward Wafer-Scale Optical Interconnects | NTT, Inc. | Transfer-printed III-V laser/PD on SiN waveguide; toward wafer-scale optical interconnects |
| `W2A.15` | Greater Than 100mW Coupled Power From O-Band Quantum Dot Laser to Silicon Nitride Waveguides Through Micro-Transfer Print Integration | Poster Session I | High-power QD laser-to-SiN coupling through micro-transfer printing |
| `Th1D.1` | Photonics Heterogeneous Integration (PHI) of Thin-Film Lithium Niobate and Hydrogen-Free Silicon Nitride on a 200-mm Silicon Photonics Platform | Institute of Microelectronics (IME); National Semiconductor Translation and Innovation Centre | 200-mm die-to-wafer TFLN + SiN heterogeneous integration |
| `M1D.4` | Wafer-Scale Heterogeneously Integrated Self-Injection-Locked Lasers | LiDAR Systems and Narrow Linewidth Lasers | 100-mm wafer-scale SiN heterogeneous laser platform with high yield for mass production |
| `W2A.4` | Wafer-Scale, Ultra-Low-Loss and Polarization-Insensitive Si3N4 Photonic Integrated Circuits | Poster Session I | Wafer-scale SiN PIC platform and component library for low-loss, polarization-insensitive routing |
| `Th3F.1` | Germanium Photodetectors With >100 GHz Bandwidth and >1.1 A/W Responsivity on 200-mm Silicon Photonics Platform | Photodetectors | Uniform wafer-scale Ge PD performance on 200-mm SiPh with >100 GHz bandwidth |
| `M2B.2` | A 280 Gbps Optical Transceiver With Monolithically Integrated 110 GHz Silicon Microring Modulators and 110 GHz Germanium Photodetectors | Fudan University; Zhangjiang Laboratory | 300-mm SiPh MRM + Ge PD monolithic link; scale-up through CMOS-line manufacturing |
| `W3E.4` | A 6.4 Tbps Optical Transmitter With Low-Loss and High-Uniformity Inverse-Designed Multiplexer on a 300-mm CMOS Platform | Zhangjiang Laboratory; Fudan University | 6.4T integrated transmitter on 300-mm CMOS/SiPh platform |
| `Th4A.6` | TFLN-Based Wafer-Level Co-Packaged Optics Engine for Ultrahigh-Bandwidth Electro-Optical Modulation | Shanghai Jiao Tong University | Wafer-level TFLN EOM + EIC CPO engine with >100 GHz bandwidth |
| `Tu2D.2` | 3D Hybrid Bonded EIC-PIC Integration and Packaging Technologies | University of California Davis | 3D hybrid bonding for EIC/PIC integration and advanced packaging |

## Scale-Up: CPO, Chiplets, Glass, Fan-Out, and Fiber Attach

| Code | Presentation / paper | Affiliation(s) | Scale-up angle |
|---|---|---|---|
| `Th3C.1` | Chiplet-to-Chiplet All-to-All Interconnecting Photonic Interposer Using AWGRs With 3D Ultrafast-Laser-Inscription | UC Davis | 3D photonic interposer with AWGRs for chiplet-to-chiplet all-to-all links |
| `Th3C.2` | Integrated Glass Waveguide Substrate With Surface Coupled Photonic Chips for Massive Scaling of CPO | Corning Optical Communication GmbH & Co. KG; Corning Technology Center Korea | Glass substrate with embedded waveguides/electrical interconnects for scalable SiPh assembly |
| `Th3C.3` | Vertical Optical Coupling Tapers for Co-Packaged Optics With Multimode Fiber and High-Speed Photodetectors | Co-packaged Optics and Advanced Packaging Techniques | Multimode-fiber-array coupling to high-speed PDs with >95% coupling and +/-25 um tolerance |
| `Th3C.4` | High Performing Photonics Systems - CPO, Towards Photonics Chiplets | Fraunhofer IZM | Invited packaging/system paper on CPO, interposers, and photonic chiplets for 100T+ platforms |
| `Th3C.5` | Ultra-Low Loss Compact SiP Polarization Compensator for CPO With an ELS | McGill University; Ericsson | External-laser-source support block for practical CPO transceivers |
| `Th1D.3` | Glass Waveguides With 0.01 dB/cm Bend Loss for High-Speed, High-Density Optical Fan-Out for Co-Packaged Optics | Photonics Platforms, Fabrication Methods, and Low Loss Materials | Compact glass fan-out for fiber-to-chip pitch conversion with low coupling loss |
| `Th1D.4` | Low Loss and Mass Producible Multi-Core Fiber Fan-in/Fan-Out Device Based on Laser Direct Writting Glass Waveguides | Photonics Platforms, Fabrication Methods, and Low Loss Materials | Laser-written glass fan-in/fan-out for MCF, with manufacturability-oriented writing speed |
| `Th1D.8` | Scalable Freeform fan-in 48 Waveguide Array in Glass | Photonics Platforms, Fabrication Methods, and Low Loss Materials | 3D glass waveguide fan-in converting 1D to 2D arrays for dense optical routing |
| `Th2A.8` | Meta-Lens for co-Package Optics and Fiber Array Unit Coupling | Poster Session II | Detachable FAU-to-SiPh coupling with metalens-assisted alignment tolerance |
| `W4B.2` | All-Dielectric Integrated Microlens Coupler for Scalable and Efficient Photonic I/Os | University of California Berkeley | Wafer-scale SiON microlens coupler for robust, low-loss fiber-to-chip I/O |
| `W1B.3` | >+25-dBm x 8-Channel SOA-Integrated DFB-LD-Based TOSA for CPO External Laser Sources | Laser Prototypes and Packaging | High-power 8-channel ELS source for CPO modules |
| `M1B.1` | 9dB Link Margin Gain Using a Quantum Dot SOA in an 8-Wavelength DWDM 50Gbps NRZ CPO Link | Data Center WDM and Comb | QD SOA in DWDM CPO link to reduce transmitter power / improve link margin |

## Scale-Up: Optical Engines, WDM Parallelism, and High-Radix Fabrics

| Code | Presentation / paper | Affiliation(s) | Scale-up angle |
|---|---|---|---|
| `M4B.2` | A 256 Gb/s DWDM Optical I/O in a 3D-Stacked EIC/PIC Silicon Photonics Platform | NVIDIA Corporation | 3D-stacked 7-nm EIC / 65-nm SiPh DWDM optical I/O; high areal bandwidth density |
| `M4B.5` | Highly-Integrated 16-Channel Silicon-Photonics Optical Engine Enabling PAM6 Transmission With BER < 1E-9 | Ciena Corporation | 16-channel SiPh optical engine exceeding 2 Tb/s aggregate |
| `Th4A.7` | 1.6T (8x200Gb/s) 2xFR4 Silicon Photonic IMDD Transceiver With Monolithically Integrated Ultra-Low Crosstalk and Wideband Multiplexer | HGGenuine Optics Tech Co., Ltd. | Monolithic 1.6T SiPh transceiver in OSFP module with compact wideband MUX |
| `Th1C.4` | 1.6 Tb/s Monolithic InP Transmitter PIC With DFB, MZM, and SOA Arrays | Nokia Corporation | Eight-channel monolithic InP transmitter PIC with sources, modulators, and SOAs |
| `Th4A.1` | 4-ch x 400-Gbps PAM4 O-Band Membrane InGaAlAs EA-DFB Laser Array on a Si Photonics Platform | Device Innovation Center, NTT, Inc. | Heterogeneous 4-channel membrane EA-DFB laser array on Si; high shoreline density |
| `Tu3J.3` | High-Bandwidth-Density Uncooled EML Array for up to 770 Gb/s/mm and 11 km Fiber Reach | High-Speed EAMs and EMLs | Compact uncooled EML array with integrated RF routing and high bandwidth density |
| `M1B.2` | An 8x256 Gbps DWDM Silicon Photonic Microring Transmitter Using a 200 GHz Spaced Quantum-Dot Comb Laser | Data Center WDM and Comb | QD comb + SiPh MRM transmitter using WDM parallelism for 8x256G |
| `M2A.4` | A 16x128 Gbps DWDM Wavelength-Locked Silicon Photonic Microring Transmitter Enabled by a Quantum-Dot Comb Laser | Silicon Photonics Modulators | 2 Tb/s/fiber DWDM SiPh transmitter using 16 wavelengths and wavelength locking |
| `Th2A.13` | High-Efficiency Silicon Nitride Microcombs for Co-Packaged Optics | Poster Session II | 24-channel high-efficiency SiN microcomb for scalable datacenter/CPO interconnects |
| `W2A.12` | A 4 Tbps 16-Channel DWDM Transmitter Using Extended-Depletion Silicon Photonic Microdisk Modulator Array | Xi'an Institute of Optics and Precision Mechanics | 16-channel SiPh microdisk modulator array, 4 Tb/s aggregate |
| `M4F.1` | 1024x1024 All-to-All Interconnect Thin-CLOS-LION System Using 64 Lambda Routing on Athermal 64x64 ULCF AWGRs | University of California Davis; Enablence, Inc | 1024x1024 all-to-all optical interconnect using AWGR wavelength routing |
| `M2D.3` | High Port Count Silicon Photonic MEMS Circuit Switch | nEye.ai | SiPh MEMS OCS as scalable, low-power switching fabric |
| `M3F.3` | A 4,096x4,096 Strictly Non-Blocking Optical Circuit Switch Delivering 819.2 Tb/s via Space-and-Wavelength Routing | Nagoya University; Toyota Technological Institute | High-radix OCS fabric scaling with space x wavelength routing |
| `W4C.7` | Si-SiN Focal Plane Switch Array Chip Enabled Parallel Solid-State LiDAR | Wireless Integrated Sensing and Communications | Parallel switch-array photonic architecture; adjacent evidence for large switch-array integration |

## Papers by Watched Google Scholar Alert Authors

| Code | Presentation / paper | Affiliation(s) | Matched alert author(s) |
|---|---|---|---|
| `Th1H.2` | Silicon-Organic Hybrid (SOH) Racetrack Modulators for 200 Gbit/s PAM4 Signaling With Ultra-low Drive Voltages | Karlsruher Institut fur Technologie | Christian Koos |
| `Th3J.5` | Silicon-Organic Hybrid (SOH) IQ Modulators With Sub-1 v pi-Voltages Operating at 200 GBd 16QAM and 144 GBd 64QAM | Karlsruhe Institute of Technology (KIT); SilOriX GmbH | Christian Koos |
| `M2J.5` | Characterization of Multi-Path Interference of 10-km Hollow Core Fiber Using Swept Wavelength Interferometry | Nokia Bell Labs; Yangtze Optical Fibre and Cable Joint Stock Ltd Co; nokia | David Neilson |
| `M3E.1` | Advances in Pump Delivery and Recycling for High-Efficiency Multicore Erbium-Doped Fiber Amplifiers | Nokia Bell Labs; Sumitomo Electric Industries, Ltd. | David Neilson |
| `M4J.5` | Coexistence Demonstration of Reflective OFDR Sensing and Commercial Transceivers in a Submarine Testbed | Nokia Bell Labs | David Neilson |
| `M4J.7` | Real-Time Coherent OFDR Over Live Networks: From Access to Subsea | Nokia Bell Labs | David Neilson |
| `Th1K.1` | Real-Time, 1-m Resolution Measurement of the Optical Distribution Network of a 21-km, 1:32 PON With a Coherent Optical Frequency Domain Reflectometer | Nokia Bell Labs | David Neilson |
| `Th4B.5` | 450 Tb/s GMI, 42.4 THz, OESCL-Band Transmission Over a Field-Deployed Fiber | NICT; University College London; Aston University; Lightera Laboratories | David Neilson |
| `Th4C.7` | High-Resolution Trans-Oceanic Distributed Acoustic Sensing Enabled by a Bi-Directional Sensor Implementation | Nokia Bell Labs; Seismics Unusual LLC; Leidos Inc; Valey Kamalov LLC | David Neilson |
| `Tu2C.2` | Distributed and Dynamic AI Agent Collaboration Over Optical Transport for Network Orchestration and Monitoring | Nokia Bell Labs | David Neilson |
| `W3A.3` | Impact of Tight Channel Spacing With Fiber Switching in WDM Networks Using 800ZR+ Interfaces | Nokia Bell Labs, USA; NOKIA | David Neilson |
| `W1B.4` | A 53-Gbaud NRZ/PAM4 x 8-Channel 1060-nm Single-Mode VCSEL-Based Ultra-Compact and High-Energy-Efficient CPO Transceiver for Full-Reach Datacenter Links | Furukawa Denki Kogyo Kabushiki Kaisha; Fuji Film Business Innovation Kabushiki Kaisha; Institute of Science Tokyo | Fumio Koyama |
| `W4E.3` | 200-Gb/s 1060-nm Single-Mode Coupled-Cavity VCSEL Enabling Modal-Dispersion Free >50-GHz Bandwidth Over 500-m Multimode Fiber | Institute of Science Tokyo | Fumio Koyama |
| `W4E.5` | O-Band Membrane Surface-Emitting Laser on a Si Substrate Demonstrating 100-Gbps PAM-4 Operation | Device Technology Labs., NTT, Inc.; Institute of Integrated Research, Institute of Science Tokyo | Fumio Koyama |
| `Tu3C.2` | Ring Resonator-Based Dynamic Controller for Precise Wavelength Separation of a DWDM Laser Source | Intel Corporation | Haisheng Rong |
| `M1E.4` | On-Chip Programmable MZI-Based Fourier Synthesizer for Ultra-Broadband Kerr-Comb Equalization | Columbia University | Michal Lipson |
| `Th1F.5` | High-Power Kerr Comb Source for Data Communications | Columbia University | Michal Lipson |
| `W4B.2` | All-Dielectric Integrated Microlens Coupler for Scalable and Efficient Photonic I/Os | University of California Berkeley | Ming C. Wu |
| `M4B.5` | Highly-Integrated 16-Channel Silicon-Photonics Optical Engine Enabling PAM6 Transmission With BER < 1E-9 | Ciena Corporation | Peter Winzer |
| `W1D.7` | Co-Design of Electronic and Photonic Systems for Future LPO, NPO, and CPO | Ciena Corporation | Peter Winzer |
| `M1E.5` | Ultra-Broadband Bidirectional Spectrometer for Parallel Detection | Cambridge University; Glitterin Technology | Richard Penty |
| `M2D.2` | Curved Tunable Directional Couplers Empower Ultralow-Crosstalk, Low-Loss Optical Switch Fabrics | University of Cambridge; Universiteit Gent; University of California, Los Angeles | Richard Penty |
| `M3F.5` | Training-Phase-Aware Optical Circuit Switching Reconfiguration for Large Language Model | University of Cambridge; Interuniversitair Micro-Elektronica Centrum | Richard Penty |
| `W2A.6` | Encoder-Decoder Codesign of Lightweight 3D Surface Profiler via Integrated Photonic Sampler | University of Cambridge; GlitterinTech Limited | Richard Penty |
| `W2A.7` | Quasi-Wavelength-Agnostic Photonic Coupler Using 3D-Nanoprinting | University of Cambridge | Richard Penty |
| `M1F.7` | Renyi Divergence-Based Nonlinear OLT-Side Tomlinson-Harashima Precoding for 100G FTN PON | The Chinese University of Hong Kong; South China University of Technology; University of California Davis; Politecnico di Torino | S. J. Ben Yoo |
| `M4F.1` | 1024x1024 All-to-All Interconnect Thin-CLOS-LION System Using 64 Lambda Routing on Athermal 64x64 ULCF AWGRs | University of California Davis; Enablence, Inc | S. J. Ben Yoo |
| `Th3C.1` | Chiplet-to-Chiplet All-to-All Interconnecting Photonic Interposer Using AWGRs With 3D Ultrafast-Laser-Inscription | UC Davis | S. J. Ben Yoo |
| `Tu2D.2` | 3D Hybrid Bonded EIC-PIC Integration and Packaging Technologies | University of California Davis | S. J. Ben Yoo |
| `Th1F.6` | A Photonic Integrated Mode-Locked Laser Based on Dispersion-Managed Mode-Locking Architecture | Ecole polytechnique federale de Lausanne; Helmholtz-Zentrum Dresden-Rossendorf | Tobias J. Kippenberg |
| `Th2A.21` | Evaluation of an Erbium Doped Waveguide Amplifier RF Performance in Microwave Photonics Applications | Consorzio Nazionale Interuniversitario per le Telecomunicazioni; Scuola Superiore Sant'Anna; Ecole polytechnique federale de Lausanne | Tobias J. Kippenberg |
| `W1A.5` | Lithium-Tantalate-on-Fused Silica Mach-Zehnder Modulators | Ecole Polytechnique Federale de Lausanne; Karlsruher Institut fur Technologie | Tobias J. Kippenberg; Christian Koos |
| `W1A.7` | A 1.6 Tbit/s WDM Integrated Photonic IMDD Transmitter on Thin-Film Lithium Tantalate | Swiss Federal Institute of Technology; Karlsruher Institut fur Technologie; Shanghai Institute of Microsystem and Information Technology | Tobias J. Kippenberg; Christian Koos |

## Shortlist

The densest must-read cluster for high-speed material/platform demos is: `Th4B.2`, `Th4B.3`, `Th4A.1`, `Th4A.2`, `Th4A.6`, `Th4A.7`, `W4J.4`, `Th1D.1`, `M4D.2`, and `M4D.4`.

## Audit: Missing High-Confidence Matches

These were found in the cached metadata/text and are not in the tables above. I would add the first two groups before treating the list as comprehensive.

### High Baud / Lane-Rate Device and Module Demos

| Code | Presentation / paper | Affiliation(s) | Why it matches |
|---|---|---|---|
| `M2B.3` | A 336Gb/s/Lane 2.47pJ/b Integrated Transmitter With Silicon Photonic TW-MZM and BW-Boost High-Linearity Driver | Institute of Semiconductors, CAS; Zhangjiang Laboratory | Hybrid SiPh TW-MZM + SiGe BiCMOS driver; 336 Gb/s/lane PAM8 |
| `M2B.6` | Technologies for 400G/Lane IM/DD Interconnects | Coherent Corp. | Invited/survey style paper explicitly comparing SiPh, InP, TFLN, BTO, SOH, and plasmonic routes to 400G/lane |
| `Tu3J.5` | 360 Gbps PAM4 Differentially Driven EML With 100 GHz 3-dB Bandwidth Dual Series-Connected EAMs for Next-Generation 3.2 Tbps Data Center Transceivers | Mitsubishi Electric Corporation | InP/EML high-speed lane demo; 360 Gbps / 180 Gbaud PAM4, >100 GHz BW |
| `Tu3J.6` | 400G per Lane Differential Drive Electroabsorption Modulated Lasers (EML) With 99GHz 6-dB EO BW for Next Generation 3.2T IM-DD Applications | Broadcom | InP differential-drive EML; 320 Gb/s PAM4 and 413 Gb/s PAM6 at 160 GBd |
| `Tu3J.4` | Differential Drive EML With Tandem Modulator Structure for 200G/Lane and Beyond Applications | Mitsubishi Electric Corporation | InP EML; 113 Gbd PAM4 for 200G/lane, positioned toward 400G/lane |
| `M2B.2` | A 280 Gbps Optical Transceiver With Monolithically Integrated 110 GHz Silicon Microring Modulators and 110 GHz Germanium Photodetectors | Fudan University; Zhangjiang Laboratory | 300-mm SiPh MRM + Ge PD link; 280 Gbps PAM4 |
| `Th2A.11` | A 280 Gbps PAM6 Silicon Photonic Tabbed-Electrode Mach-Zehnder Modulator With Co-Optimized Modulation Efficiency and Electro-Optic Bandwidth | Xi'an Institute of Optics and Precision Mechanics, CAS | Silicon MZM; 66 GHz 1-dB EO BW and 280 Gbps PAM6 |
| `Th2A.14` | Record-High 90-GHz Silicon Microring Modulator With Compact RLC Modeling and 224-Gb/s PAM4 Operation Toward Co-Packaged Optics Integrations | Taiwan Semiconductor Research Institute; National Tsing-Hua University | Si MRM; 90 GHz EO BW, 224 Gb/s PAM4 |
| `W3E.5` | Toward 400 G/Lane Silicon Differential-Drive Mach-Zehnder Modulator With >80 GHz Bandwidth for Optical Interconnects | Advanced Micro Foundry Pte Ltd | Silicon differential-drive MZM; >80 GHz BW, toward 400G/lane |
| `W2A.14` | High-Bandwidth Serpentine Segmented Silicon Photonic Mach-Zehnder Modulator for 192 Gbaud Transmission | Chinese Academy of Sciences Xi'an Institute of Optics and Precision Mechanics | SiPh segmented MZM; 192 Gbaud transmission |
| `Th2A.15` | SiGe Photodetector Using a Tapered Ge Design for 400 Gbps Optical Links | McGill University; Marvell | CMOS-compatible Ge-on-Si PD; up to 400 Gbps PAM8 receiver operation |
| `W2A.42` | 300-Gb/s/lambda PAM4 IM/DD Link Enabled by GeSi Electro-Absorption Modulator and BU-GRU Equalization | KTH; University of Copenhagen; Riga Technical University; Technical University of Denmark; RISE | GeSi EAM high-speed IM/DD link; 300 Gb/s/lambda PAM4 |

### Heterogeneous / Hybrid Integration Additions

| Code | Presentation / paper | Affiliation(s) | Integration angle |
|---|---|---|---|
| `M4D.3` | High L-Band Responsivity of Compact InP-on-Si Coherent-Receiver PICs via Chip-on-Wafer Bonding | Photonics Electronics Technology Research Association; University of Tokyo | Directly in scope for InP-on-Si / SOI chip-on-wafer integration |
| `M4D.5` | 82-mm-Long Optical Link Using Micro-Transfer-Printed Directly Modulated Membrane Laser and Photodetector on SiN Waveguide: Toward Wafer-Scale Optical Interconnects | NTT, Inc. | Transfer-printed membrane laser/PD on SiN waveguide |
| `Tu2D.2` | 3D Hybrid Bonded EIC-PIC Integration and Packaging Technologies | University of California Davis | 3D hybrid bonding for SiPh PIC + CMOS EIC integration |
| `Th3C.4` | High Performing Photonics Systems - CPO, Towards Photonics Chiplets | Fraunhofer IZM | Invited/system paper on CPO, photonic chiplets, interposers, 2.5D/3D/heterogeneous integration |
| `Th3C.5` | Ultra-Low Loss Compact SiP Polarization Compensator for CPO With an ELS | McGill University; Ericsson | SiP CPO support component for external laser source polarization handling |
| `Th4A.5` | Fully Integrated 1064 nm Transmitters With Widely Tunable GaAs Lasers and >100-GHz Thin-Film LiNbO3 Modulators | Nexus Photonics; Northeastern University; Keysight Technologies Inc. | TFLN modulator integration with tunable GaAs lasers; high-speed non-datacom wavelength demo |

### 1.6T+ / System-Level Additions

| Code | Presentation / paper | Affiliation(s) | Scale demonstrated |
|---|---|---|---|
| `W1E.3` | Real-Time 1.2 Tb/s S-Band Silicon Photonics Transceiver Operating at 6-THz Tunable Bandwidth | ZTE Corporation; ZTE Photonics Technology Japan | Real-time 1.2 Tb/s S-band SiPh transceiver |
| `Th4B.6` | Multi-Vendor Interoperable 800G Optical Communications Over 1602-km Through 800G-OpenROADM, 800-ZR and 800GBASE-2xFR4 Fiber Links | Orange Research; Ciena; Cisco/Acacia; Nokia; Coherent; EXFO | 800G multi-vendor interoperability; relevant adjacent stepping stone to 1.6T ecosystems |
| `W3J.4` | 153.8 Tb/s O-Band Coherent Transmission Over SMF With Low-Complexity DSP | UCL; Corning; Lightera Labs; NICT | 153.8 Tb/s O-band WDM coherent transmission |
| `Th4C.8` | Carrier/Clock-Shared Comb-Based Superchannel With <1-ps Timing Error Enabling Baud-Rate Sampling Coherent Reception for Scale-Across AIDCs | Peking University; Shanghai Jiao Tong University | 18 wavelength x 384 Gb/s superchannel for AIDC distributed training |
| `W2A.12` | A 4 Tbps 16-Channel DWDM Transmitter Using Extended-Depletion Silicon Photonic Microdisk Modulator Array | Xi'an Institute of Optics and Precision Mechanics | 4 Tb/s SiPh DWDM transmitter; 16-channel integrated microdisk modulator array |
| `W3E.4` | A 6.4 Tbps Optical Transmitter With Low-Loss and High-Uniformity Inverse-Designed Multiplexer on a 300-mm CMOS Platform | Zhangjiang Laboratory; Fudan University | 6.4 Tb/s integrated optical transmitter on 300-mm CMOS/SiPh platform |

## Audit: Corrections / Caveats

| Location | Current text | Suggested correction |
|---|---|---|
| Line 13 | `Th1C.3` is described as "TFLN MZM; 400G/lane..." | Keep, but note it is a broader linear-drive optics paper and discusses TFLN, InP EML/EAM-on-Si, etc.; it is not only a TFLN-device paper. |
| Line 21 | `W3E.6` says "Co-packaged InP MZM + driver" | More precise: O-band driver-modulator engine with InP MZM and integrated driver; 180 GBaud PAM4. |
| Line 22 | `W1A.3` says "TFLN on 200-mm Si; 110 GHz BW" | Correct, but title says "200-mm Silicon Substrate"; if this is used as an integration example, distinguish it from the stronger PHI/D2W `Th1D.1` result. |
| Line 30 | `Th1D.1` title abbreviates "TFLN" and "SiN" | Exact metadata title: "Photonics Heterogeneous Integration (PHI) of Thin-Film Lithium Niobate and Hydrogen-Free Silicon Nitride on a 200-mm Silicon Photonics Platform". |
| Line 35 / 46 | `Th4B.3` title shortened to "BTO..." | Exact metadata title uses "Barium Titanate Enabling Net 1.6T (4x448 Gbps PAM4) on a Silicon Photonics Platform". |
| Line 45 | `Th4A.7` title is shortened | Exact metadata title: "1.6T (8x200Gb/s) 2xFR4 Silicon Photonic IMDD Transceiver With Monolithically Integrated Ultra-Low Crosstalk and Wideband Multiplexer". |
| Line 53 | `Th2A.46` omits "simultaneous 100 GHz" | Exact metadata title includes "25 Simultaneous 100 GHz Comb Channels". |
| Lines 58-59 | `M1H.1` and `M3F.3` are included in 1.6T+ | They are valid high-throughput demos, but less relevant to lane-rate/material/SiPh integration than the missing SiPh/EML/CPO entries above. |
