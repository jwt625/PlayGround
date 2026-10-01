---
paper_id: xu2026a
source_url: https://doi.org/10.1364/ofc.2026.w1a.4
doi: 10.1364/ofc.2026.w1a.4
license: 
extracted_on: 2026-10-01
extraction_method: pre-extracted text copied from a local corpus; no PDF and no figure images available; figures and tables must be treated as unavailable
---

# Sub-V-driven 110-GHz O-band Electro-optic Modulator on Thin-film Litihum Tantalate

Mengyue Xu\*,†, Yang Lan†, Jinze Shi and Di Liang\*

Department of Electrical and Computer Engineering, University of Michigan, Ann Arbor, MI 48109 USA

† Equal contributions

\*mengyue.xu01@gmail.com, liangdi@umich.edu

**Abstract:** We demonstrate O-band Mach–Zehnder modulators on thin film lithium tantalate with  $V_{\pi}$  of 1 V from 10 Hz to 10 kHz and electro-optic bandwidth beyond 110 GHz, enabled by localized silicon substrate removal.

#### 1. Introduction

Lithium tantalate offers an electro-optic (EO) coefficient (27.7 pm/V at 1310 nm [1]) comparable to lithium niobate, while providing a higher optical damage threshold, smaller birefringence, and lower dielectric loss tangent, which benefit passive optical performance and reduce RF loss. More importantly, thin-film lithium tantalate (TFLT) modulators exhibit reduced DC bias drift, a critical concern for lithium niobate. Counterparts, TFLT modulators operating in the C-band and O-band have shown notable progress [2–7]. Reference [4] reported a 110 GHz bandwidth with a  $V_{\pi}$  of 4.8 V using silver electrodes to lower conductor loss. However, the device still experienced larger than 5 dB/cm of microwave attenuation at 70 GHz due to dielectric loss in silicon and a narrow electrode gap. More recent work used complete backside silicon removal to cut microwave loss but required precise double-sided mask alignment. In addition, large group delay mismatch due to the difficulty of controlling deep backside Si hole dimensions limited the electro-optic bandwidth to 55 GHz [2].

We demonstrate a 15-mm-long O-band Mach–Zehnder modulator (MZM) on the TFLT platform that combines optimized EO design with locally silicon removal underneath the transmission line, achieving 4.8 dB/cm microwave attenuation at 110 GHz. The low-frequency  $V_{\pi}$  shows minimal variation, ranging from 1.045 V to 1.166 V across 10 Hz to 10 kHz. The 3-dB EO bandwidth (BW) exceeds 110 GHz, with a return loss less than -20 dB across the band. To the best of our knowledge, our device shows the highest reported value of  $BW/V_{\pi}^2$  (100.7 GHz/V<sup>2</sup>) on the TFLT platform.

## 2. Results

![](_page_0_Figure_11.jpeg)

Fig. 1. (a) Photograph of the experimental setup with the TFLT chip. (b–c) Schematic and cross section of the locally silicon-removed TFLT MZM. (d) Simulated fundamental TE mode distribution. (e) Simulated RF electric field distribution.

A fabricated MZM chip on the characterization stage is shown in Fig. 1(a). We designed an asymmetric MZM with a  $2 \times 2$  multimode interference (MMI) coupler and capacitively loaded traveling-wave electrodes (CL-TWEs), as shown in Fig. 1(b). The device was fabricated on a commercially available x-cut TFLT wafer from NANOLN with 400-nm-thick lithium tantalate on a 3- $\mu$ m-thick buried silicon dioxide layer over a silicon substrate. Optical waveguide circuits were patterned by electron-beam lithography on a silicon dioxide hard mask, then transferred into the TFLT by argon ion milling. Gold electrodes with 1  $\mu$ m thickness were formed by a lift-off process. To reduce microwave loss, periodically distributed through-vias next to the TFLT waveguides were etched through

![](_page_1_Figure_2.jpeg)

Fig. 2. (a) A scanning electron microscope image of the LT waveguide. (b) Measured optical transmission of the asymmetric MZM, indicating an on-chip loss of 1.2 dB and an extinction ratio greater than 35 dB.

the lithium tantalate and buried oxide using ICP etching, followed by local removal of the silicon underneath the electrodes (Fig. 1(c)) with xenon difluoride-based isotropic dry etching. Note that we use a positive e-beam resist to pattern only the two 200 nm-deep side trenches that define the rib waveguide in the modulation region, allowing the electrodes to sit on the intact TFLT membrane, as shown in Figs. 1(c–d). This vertical offset relative to the fundamental TE mode allows the metal electrodes to be placed closer to the waveguide without incurring significant optical absorption. In our design, the electrode spacing is 4  $\mu$ m and the waveguide width is 2  $\mu$ m, yielding a simulated optical loss of 0.01 dB/cm and a  $V_{\pi}L$  of 1.6 V·cm. as shown in Figs. 1(d-e).

The in-house fabricated MZM has a 15 mm long modulation region. The SEM image in Fig. 2(a) shows smooth LT waveguide sidewalls. The measured optical transmission around 1310 nm in Fig. 2(b) indicates an MZM insertion loss of 1.2 dB and an extinction ratio up to 35 dB after normalizing the impact of the grating couplers. The low loss and high extinction ratio are prerequisites for high optical signal-to-noise ratio and a large dynamic extinction ratio in future eye diagram measurements once the instrument is available.

![](_page_1_Figure_6.jpeg)

Fig. 3. (a) Measured electrical reflection  $S_{11}$  for a 15 mm long modulator. (b) Measured microwave attenuation extracted from S parameters.

S-parameters of the 15 mm-long CL-TWE was measured using a vector network analyzer (VNA) with millimeter-wave extenders up to 110 GHz. As shown in Fig. 3(a), the electrical–electrical (EE)  $S_{11}$  remains below -20 dB across the measured band. The extracted microwave attenuation is only 3.2 dB/cm at 70 GHz and 4.8 dB/cm at 110 GHz (Fig. 3(b)), attributable to our CL-TWE design and local silicon removal, and is substantially lower than prior reported values [5,7].

Next, we measured the low-frequency  $V_{\pi}$  from 10 Hz to 10 kHz using a triangular-wave drive to assess dynamic stability. In many lithium niobate modulators, the low-frequency  $V_{\pi}$  is inconsistent and difficult to measure because of DC drift [8]. As shown in Fig. 4(a), our TFLT device maintains a nearly constant  $V_{\pi}$  with an average of 1.089 V (1.045 V at 10 Hz), corresponding to  $V_{\pi}L = 1.63 \text{ V} \cdot \text{cm}$  (1.57 V · cm at 10 Hz), in agreement with simulation.

To evaluate the electro-optic bandwidth of the fabricated MZM, we used a VNA and a 110 GHz high-speed photodetector (HHI) for measurement. Fig. 4(b) exhibits a 3 dB EO response extending beyond 110 GHz with a reference to the value at 2 GHz. Comparing our bandwidth-voltage performance with stat-of-the-art TFLT results in Fig. 5, to our best knowledge, we demonstrate the highest  $BW/V_{\pi}^2$ : 100.7 GHz/V<sup>2</sup>, a critical figure of merit to simultaneously gauge operation speed and voltage for AI-driven optical interconnect applications.

![](_page_2_Figure_2.jpeg)

Fig. 4. (a) Measured transmission of the TFLT MZM driven by low modulation frequencies from 10 Hz to 10 kHz at the wavelength of 1310 nm. (b) Measured EO  $S_{21}$  response, showing 3-dB EO bandwidth beyond 110 GHz.

![](_page_2_Figure_4.jpeg)

Fig. 5. Comparison of voltage-bandwidth performance on TFLT [2–7].

### 3. Conclusion

We present a TFLT MZM with local silicon substrate removal that enables superior modulation bandwidth to device drive voltage performance. The device with 15 mm-long modulation section exhibits low on-chip loss of 1.2 dB, extinction ratio above 35 dB, return loss less than -20 dB up to 110 GHz, a stable  $V_{\pi}$  of 1.089 V, and a 3-dB EO bandwidth exceeding 110 GHz. These metrics indicate strong suitability for high-speed and energy-efficient IM/DD modulation. Large-signal characteristics are pending due to the availability of necessary instrument at present and will be thoroughly measured and reported by the conference. The demonstrated TFLT modulators can be further integrated with low loss (de-)multiplexers and extended to coherent modulators, enabling a viable path toward DC-drift-free coherent transmitters at the 10 Tb/s level, and many other emerging applications.

# <span id="page-2-0"></span>References

- 1. J. L. Casson, K. T. Gahagan, D. A. Scrymgeour, R. K. Jain, J. M. Robinson, V. Gopalan, and R. K. Sander, "Electro-optic coefficients of lithium tantalate at near-infrared wavelengths," J. Opt. Soc. Am. B 21, 1948–1952 (2004).
- <span id="page-2-1"></span>2. H. Wang, A. Cui, B. Chen, Z. Ruan, C. Guo, K. Chen, and L. Liu, "Thin-film lithium tantalate modulator operating at high optical power," ACS Photonics 12, 5345–5351 (2025).
- 3. C. Wang, Z. Li, J. Riemensberger, G. Lihachev, M. Churaev, W. Kao, X. Ji, J. Zhang, T. Blesin, A. Davydova *et al.*, "Lithium tantalate photonic integrated circuits for volume manufacturing," Nature **629**, 784–790 (2024).
- <span id="page-2-3"></span> C. Wang, D. Fang, J. Zhang, A. Kotz, G. Lihachev, M. Churaev, Z. Li, A. Schwarzenberger, X. Ou, C. Koos, and T. J. Kippenberg, "Ultrabroadband thin-film lithium tantalate modulator for high-speed communications," Optica 11, 1614–1620 (2024).
- <span id="page-2-4"></span>5. M. Lin, Z. Li, J. Riemensberger, and T. J. Kippenberg, "Copper damascene process-based high-performance thin film lithium tantalate modulators," in *CLEO* 2025, (Optica Publishing Group, 2025), p. PD103\_6.
- 6. M. Zhu, F. Huang, D. Liu, W. Wang, A. Gao, W. Zhao, S. Zhao, D. Lian, C. Gao, Z. Yu *et al.*, "An 8× 200 gbps wavelength-division multiplexing transmitter using lithium tantalate," PhotoniX **6**, 27 (2025).
- <span id="page-2-2"></span>7. J. Cai, A. Kotz, H. Larocque, C. Wang, X. Ji, J. Zhang, D. Drayss, X. Ou, C. Koos, and T. J. Kippenberg, "Heterogeneously integrated lithium tantalate-on-silicon nitride modulators for high-speed communications," (2025).
- <span id="page-2-5"></span>8. M. Xu, M. He, H. Zhang, J. Jian, Y. Pan, X. Liu, L. Chen, X. Meng, H. Chen, Z. Li *et al.*, "High-performance coherent optical modulators based on thin-film lithium niobate platform," Nat. communications **11**, 3911 (2020).