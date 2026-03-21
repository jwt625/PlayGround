# Systematic Improvements in Transmon Qubit Coherence Enabled by Niobium Surface Encapsulation

Mustafa Bal,<sup>1, a)</sup> Akshay A. Murthy,<sup>1, a)</sup> Shaojiang Zhu,<sup>1, a)</sup> Francesco Crisa,<sup>1, a)</sup> Xinyuan You,<sup>1</sup> Ziwen Huang,<sup>1</sup> Tanay Roy,<sup>1</sup> Jaeyel Lee,<sup>1</sup> David van Zanten,<sup>1</sup> Roman Pilipenko,<sup>1</sup> Ivan Nekrashevich,<sup>1</sup> Andrei Lunin,<sup>1</sup> Daniel Bafia,<sup>1</sup> Yulia Krasnikova,<sup>1</sup> Cameron J. Kopas,<sup>2</sup> Ella O. Lachman,<sup>2</sup> Duncan Miller,<sup>2</sup> Josh Y. Mutus,<sup>2</sup> Matthew J. Reagor,<sup>2</sup> Hilal Cansizoglu,<sup>2</sup> Jayss Marshall,<sup>2</sup> David P. Pappas,<sup>2</sup> Kim Vu,<sup>2</sup> Kameshwar Yadavalli,<sup>2</sup> Jin-Su Oh,<sup>3</sup> Lin Zhou,<sup>3</sup> Matthew J. Kramer,<sup>3</sup> Florent Q. Lecocq,<sup>4</sup> Dominic P. Goronzy,<sup>5</sup> Carlos G. Torres-Castanedo,<sup>5</sup> Graham Pritchard,<sup>5</sup> Vinayak P. Dravid,<sup>5, 6, 7</sup> James M. Rondinelli,<sup>5</sup> Michael J. Bedzyk,<sup>5</sup> Mark C. Hersam,<sup>5, 8, 9</sup> John Zasadzinski,<sup>10</sup> Jens Koch,<sup>11, 12</sup> James A. Sauls,<sup>13</sup> Alexander Romannenko\*,<sup>1</sup> and Anna Grassellino\*<sup>1</sup>

(\*Electronic mail: Corresponding authors: aroman@fnal.gov, annag@fnal.gov)

(Dated: 26 January 2024)

We present a novel transmon qubit fabrication technique that yields systematic improvements in  $T_1$  relaxation times. We fabricate devices using an encapsulation strategy that involves passivating the surface of niobium and thereby preventing the formation of its lossy surface oxide. By maintaining the same superconducting metal and only varying the surface structure, this comparative investigation examining different capping materials, such as tantalum, aluminum, titanium nitride, and gold, and film substrates across different qubit foundries definitively demonstrates the detrimental impact that niobium oxides have on the coherence times of superconducting qubits, compared to native oxides of tantalum, aluminum or titanium nitride. Our surface-encapsulated niobium qubit devices exhibit  $T_1$  relaxation times 2 to 5 times longer than baseline niobium qubit devices with native niobium oxides. When capping niobium with tantalum, we obtain median qubit lifetimes above 300 microseconds, with maximum values up to 600 microseconds, that represent the highest lifetimes to date for superconducting qubits prepared on both sapphire and silicon. Our comparative structural and chemical analysis suggests why amorphous niobium oxides may induce higher losses compared to other amorphous oxides. These results are in line with high-accuracy measurements of the niobium oxide loss tangent obtained with ultra-high Q superconducting radiofrequency (SRF) cavities. This new surface encapsulation strategy enables even further reduction of dielectric losses via passivation with ambient-stable materials, while preserving fabrication and scalable manufacturability thanks to the compatibility with silicon processes.

With massive improvements in device coherence times and gate fidelity over the past two decades, superconducting qubits have emerged as a leading technology platform for quantum computing<sup>1–3</sup>. Although many of these improvements have been driven through optimized device designs and geometries, the presence of defects and impurities at the interfaces and surfaces in the constituent materials continues to limit per-

Niobium (Nb) has been widely employed as the primary material in superconducting qubits as it possesses the largest critical temperature and superconducting gap of elemen-

<sup>&</sup>lt;sup>1)</sup>Superconducting Quantum Materials and Systems Division, Fermi National Accelerator Laboratory (FNAL), Batavia, IL 60510, USA

<sup>&</sup>lt;sup>2)</sup>Rigetti Computing, Berkeley, CA 94710, USA

<sup>&</sup>lt;sup>3)</sup>Ames Laboratory, U.S. Department of Energy, Ames, IA 50011, USA

<sup>&</sup>lt;sup>4)</sup>National Institute of Standards and Technology, Boulder, CO, USA

<sup>&</sup>lt;sup>5)</sup>Department of Materials Science and Engineering, Northwestern University, Evanston, IL, 60208, USA

<sup>&</sup>lt;sup>6)</sup>The NUANCE Center, Northwestern University, Evanston, IL, 60208, USA

<sup>&</sup>lt;sup>7)</sup>International Institute of Nanotechnology, Northwestern University, Evanston, IL, 60208, USA

<sup>&</sup>lt;sup>8)</sup>Department of Chemistry, Northwestern University, Evanston, IL 60208, USA

<sup>&</sup>lt;sup>9)</sup>Department of Electrical and Computer Engineering, Northwestern University, Evanston, IL 60208, USA

<sup>&</sup>lt;sup>10)</sup>Department of Physics, Illinois Institute of Technology, Chicago, IL, 60616, USA

<sup>11)</sup> Department of Physics and Astronomy, Northwestern University, Evanston, IL 60208, IISA

<sup>&</sup>lt;sup>12)</sup>Center for Applied Physics and Superconducting Technologies, Northwestern University, Evanston, IL 60208, USA

<sup>&</sup>lt;sup>13)</sup>Hearne Institute of Theoretical Physics, Department of Physics and Astronomy, Louisiana State University, Baton Rouge, LA 70803, USA

formance and serve as a critical barrier in achieving scalable quantum systems. 4-6 Specifically, these uncontrolled defect sites can serve as sources of loss by introducing two-level systems (TLS) or nonequilibrium quasiparticles 7-10. As a result, researchers have recently begun to take a materials-oriented approach to understand and eliminate these sources of quantum decoherence in superconducting qubit devices.

<span id="page-0-0"></span>a) These authors contributed equally

tal superconductors, making thermal quasiparticle contribution to losses negligible at typical operating temperatures of  $\lesssim 50$  mK. It is also highly compatible with industrial-scale processes  $^{11}$ . Furthermore, the Fermilab superconducting radio-frequency (SRF) research group has demonstrated in prior detailed studies of 3D cavities in the quantum regime that devices processed from Nb can sustain photon lifetimes as high as 2 seconds when the surface niobium oxide hosting sources of TLS is removed.  $^{12}$ . This is  $\sim\!\!3$  orders of magnitude longer than coherence times reported in the highest-performing transmon qubits  $^{13,14}$ , making bare niobium metal an attractive base material for further improvements in 2D superconducting qubits.

These previous measurements have unambiguously identified the surface oxide that forms spontaneously on Nb under ambient conditions as the major source of microwave  $loss^{12,15-20}$ . Through 3D cavity measurements, we find that the loss tangent of this 5 nm thick oxide is  $\sim 0.1$ , which is orders of magnitude larger than the losses at the metal/substrate interface as well as those in the underlying substrate  $^{21,22}$ . As a result, the removal of this oxide has been shown to boost the photon lifetime by  $50\text{-}200\times$  in 3D Nb SRF cavities in the TLS-dominated, <1K regime. Other studies on 2D devices have since further confirmed the detrimental effect of this oxide  $^{23,24}$ .

Several recent studies have sought to mitigate losses associated with this region. Unfortunately, most methods for avoiding these losses are incompatible with integration into complex or large-scale manufacturing process flows. In one of the successful approaches, the surface oxide was removed by annealing the sample at temperatures at or exceeding 300°C<sup>12</sup>, with an almost complete elimination of the TLSinduced losses. While this thermal dissolution method is effective, sustained vacuum is required afterwards to prevent the regrowth of the surface oxide when the cavity or qubit is removed from a ultra-high vacuum (UHV) environment. An alternative oxide removal method involving HF as a wet etchant has been explored as well<sup>23,24</sup>. This process cannot be performed in vacuum, leading to rapid oxide re-growth afterwards. Furthermore, it can lead to hydrogen incorporation in the underlying niobium which can introduce resistive niobium hydrides<sup>25–27</sup>. Finally, nitrogen plasma passivation techniques have been identified as effective methods to partially suppress oxide formation<sup>28,29</sup>.

Here, we propose a new strategy based on surface encapsulation to eliminate and prevent the formation of this lossy Nb surface oxide upon exposure to air. The first method involves depositing *in situ* metal capping layers of Al and Ta onto the Nb films in UHV. The second method involves atomic layer deposition to reduce the native Nb surface oxide by reacting it with a precursor then depositing a thin metal film (TiN) that exhibits a reduced microwave loss. The third method involves milling away the oxide with Ar<sup>+</sup> ions and depositing a thin metal layer of Au with e-beam evaporation. Based on our systematic study, we observe that each of these surface passivation strategies effectively eliminate Nb<sub>2</sub>O<sub>5</sub> and yield a clear improvement in coherence times. Of these capping approaches, we find that the Ta capped Nb films exhibit the

largest improvement and lead to devices with median relaxation times of 300  $\mu$ s. Finally, we explore the scalability of such an approach by repeating fabrication of test devices with the Ta capping strategy at a commercial qubit fabrication and measurement facility and are able to replicate the results on Si substrates.

#### I. EXPERIMENTAL RESULTS

We fabricated seven sets of qubits for this study as outlined in Table I with device geometries provided in 1a. In terms of surface participation ratio, this is largest for device geometry A, followed by device geometry C, and smallest for device geometry B. Details of the fabrication procedure are provided in the Supplementary Information. Results from structural and chemical analysis of the fabricated qubits are provided in Figs. 1b and 2. Fig. 1b shows a low magnification scanning transmission electron microscopy (STEM) image of a crosssection taken from a superconducting circuit. The superconducting metal (Nb) and metal capping layer (Ta) are labeled. Chemical phase maps generated using STEM energy dispersive spectroscopy (EDS) are presented in Fig. 2a-d for the Nb films capped with different layers. We find that the capping layers are between 10-15 nm, as targeted. Additionally, the layers are spatially distinct with minimal intermixing present.

| Substrate | Film | Surface          | Foundry | Measurement Site |  |
|-----------|------|------------------|---------|------------------|--|
|           |      | Encapsulation    |         |                  |  |
| Sapphire  | Nb   | -                | PNF     | Fermilab         |  |
| Sapphire  | Nb   | Ta (Sputtering)  | PNF     | Fermilab         |  |
| Sapphire  | Nb   | Al (Sputtering)  | PNF     | Fermilab         |  |
| Sapphire  | Nb   | TiN (ALD)        | PNF     | Fermilab         |  |
| Sapphire  | Nb   | Au (Evaporation) | PNF     | Fermilab         |  |
| Silicon   | Nb   | -                | Rigetti | Rigetti          |  |
| Silicon   | Nb   | Ta (Sputtering)  | Rigetti | Rigetti          |  |

<span id="page-1-0"></span>TABLE I. List of fabricated transmon qubits.

In order to assess the efficacy of the capping layers in preventing  $Nb_2O_5$  formation, we analyze each film with time-of-flight secondary ion mass spectrometry (ToF-SIMS). This technique combines high mass resolution and sensitivity to both light and heavy elements with <100 nm spatial resolution, and has been employed extensively to identify impurities such as oxides in superconducting qubits  $^{16,30}$ . As ToF-SIMS enables identification of different oxide species based on their mass to charge (M/Z) ratios, we are able to resolve that the surface oxide is primarily composed of  $Nb_2O_5$  based on the presence of localized signal corresponding to the presence of  $Nb_2O_5$  ions in this region.

To better understand how this oxide is impacted by metal capping, we compare the  $Nb_2O_5^-$  signal counts measured from the surface of the baseline Nb sample to the  $Nb_2O_5^-$  signal counts measured at the interface between the capping layer and the Nb metal in the capped Nb samples as indicated in Fig. 2d. We note that this does not refer to total oxide quantity (as aluminum and tantalum themselves oxidize), but specifi-

![](_page_2_Figure_1.jpeg)

<span id="page-2-0"></span>FIG. 1. (a) 8 qubit chip layout consisting of 3 different geometries. (b) Low magnification annular dark field scanning transmission electron microscopy (ADF-STEM) image taken from a cross-section of a Nb transmon qubit where the Nb film is capped with a Ta metal layer. (c) Cryogenic wiring diagram. (d-e) Pulse scheme for characterization and plots. (d) Pulse for Rabi experiment used to calibrate π pulses. (e) Pulse for T<sup>1</sup> experiment.

cally to the reduction of the loss channel of particular interest, Nb2O5. The methodology is described in Fig. [S1](#page-2-0) and the results are presented in Fig. [2e](#page-3-0). We find that all of the capping strategies are highly effective in mitigating Nb2O<sup>5</sup> formation. The Ta capping is particularly effective as a 1000× decrease in measured Nb2O<sup>5</sup> <sup>−</sup> is observed with this strategy. The results also suggest that serial sputter deposition may be slightly more effective at protecting again surface oxidation compared to the other two methods. Finally, we observe the presence of a sharp interface between the Nb film and the underlying sapphire as shown in Fig. [2f](#page-3-0). In contrast to Nb films grown on silicon where alloyed regions on the order of 5nm have been observed, in Nb/sapphire structures we observe minimal intermixing present at the metal/substrate interface[31](#page-8-4) .

To most easily observe the impact of the metal capping

layer on the superconducting qubit coherence, we performed measurements using the qubit geometry with the largest surface participation ratio (geometry A). These measurements were performed on the capped devices and on baseline Nb devices that were not capped.

The qubit devices are measured inside a dilution refrigerator at a temperature of around 40 mK via dispersive readout. The cryogenic wiring diagram is shown in Fig. [1c](#page-2-0). Both qubit and readout pulses are sent through a single RF line. At the mixing chamber plate, two six-pole-single-throw microwave switches are used to direct the signals to the relevant sample and to extract the outgoing signal. After initial qubit spectroscopy a Rabi measurement (Fig. [1d](#page-2-0) is performed to determine the π-pulse length followed by T<sup>1</sup> measurements (Fig. [1e](#page-2-0)).

![](_page_3_Figure_1.jpeg)

<span id="page-3-0"></span>FIG. 2. Structural and chemical characterization of transmon qubit. (a-d) Chemical phase maps generated through STEM energy dispersive spectroscopy of the Nb films capped with Ta, Al, TiN, and Au, respectively. The Ta, Al, and Au capping layers are roughly 10 nm thick and the TiN layer is roughly 5 nm thick. (e) Plot depicting Nb<sub>2</sub>O<sub>5</sub> counts captured with ToF-SIMS. Each of the capping strategies are effective in mitigating Nb<sub>2</sub>O<sub>5</sub> formation. The Ta capping is particularly effective as a  $1000 \times$  decrease in Nb<sub>2</sub>O<sub>5</sub> counts is observed with this strategy. (f) ADF-STEM image of the metal/substrate interface. Minimal intermixing is observed between Nb and the underlying c-plane sapphire substrate.

Because the qubit energy relaxation time,  $T_1$ , is largely dependent on the material losses in the qubit device, while the dephasing time,  $T_2$ , is heavily impacted by many environmental factors such as the thermal noise, IR radiation, and cosmic rays $^{33,34}$ , we focus on qubit  $T_1$  characterization. This qubit  $T_1$ characterization is widely believed to directly reflect the loss due to TLS, which is the focus of this work. The measurement follows the standard procedure<sup>35</sup>, *i.e.*, the qubit is driven from the ground to the first-excited state by a calibrated  $\pi$ pulse. The qubit state is then read after a variable delay. The relaxation times are fitted to a single-exponential decay to extract T<sub>1</sub>. In order to probe both the typical and exceptional  $T_1$  times, we benchmark the  $T_1$  measurement by continuously collecting data for 10 hours for each qubit, as described in Ref [ 36]. Using this data set, we extract the average, standard deviation, and best T<sub>1</sub> values, which we compare across different devices.

A comparison of the  $T_1$  measured in different devices is provided in Fig. 3(a) and further summarized in Table S1. The Nb qubits capped with Ta as well as the Nb qubits capped with Au have the highest average  $T_1$  (>  $100\mu s$ ), while the baseline Nb qubits have the lowest average  $T_1$ . The average  $T_1$  value of Nb qubits capped with Al are slightly higher than that of Nb qubits capped with TiN. The improvement in av-

erage  $T_1$  for all of the capped devices suggests that reducing the native  $Nb_2O_5$  (a strong TLS host) on the surface of the Nb film improves qubit energy relaxation times. Whiskers or circles in the plot indicate the maximum and minimum  $T_1$  values observed during the greater than 10 hours measurement window per qubit [see Fig 3(d) for the best  $T_1$  decay curve]. The fluctuations of  $T_1$  over time [shown in Fig. 3(e)], and its Gaussian distribution [shown in Fig. 3(f)] have both been observed in literature  $^{36,37}$ , and are typically considered a signature of qubit lifetime limited by TLS defects residing in the materials.

To quantitatively compare TLS densities from different encapsulations, we calculate the average  $(\mu_{T_1})$  vs. standard deviation  $(\sigma_{T_1})$   $T_1$  for each of the qubits that were measured continuously for 10 hours. According to TLS theory<sup>32</sup>,  $\mu_{T_1}$  and  $\sigma_{T_1}$  can be represented by the following relations:

$$\mu_{T_1} = \alpha/(\omega N), \tag{1}$$

$$\sigma_{T_1} = \beta / (\omega N)^{3/2}, \qquad (2)$$

where  $\alpha$  and  $\beta$  are temperature-dependent constants and  $\omega$  is the qubit frequency, which is treated as constant in this case as the qubit frequencies all lie within the range of 4 and 6 GHz. N is proportional to the number of TLS present in the qubit.

![](_page_4_Figure_1.jpeg)

<span id="page-4-0"></span>FIG. 3. Qubit measurement data (a)  $T_1$  comparison of the five sets of qubit devices that were prepared on sapphire substrates. All four Nb/Ta qubits on the chip show  $T_1>100~\mu s$ , and the largest  $T_1$  measured for Nb control qubits is  $\sim 50~\mu s$ . Boxes mark the 25th percentile and the 75th percentile of the measurement distribution over the course of 10 hours of consecutive measurements. The line inside each box represents the median value, and whiskers or circles represent outliers. (b) Measured  $T_1$  values for test devices fabricated on silicon substrates. We observe a clear improvement in terms of the median  $T_1$  value following surface capping of Nb with  $T_2$ . (c) Dependence of  $T_1$  standard deviation,  $\sigma_{T_1}$ , on the average  $T_1$ ,  $\mu_{T_1}$ . Different colors correspond to the different encapsulation groups shown in (a). Dashed line shows the best fitting of  $\sigma_{T_1} \propto \mu_{T_1}^{3/2}$ , according to Ref[ 32]. It also shows that the Nb/Ta qubit has 5-10 times improvement of TLS loss compared with the Nb qubit, after converting the number of TLS into the tangent loss,  $\delta_{TLS}$ . (d) Best  $T_1$ =198  $\mu s$ . (e) Statistics for  $T_1$  consecutively measured over 10 hours, the average  $T_1$  ( $\mu_{T_1}$ ) is  $161\mu s$  and the standard deviation ( $\sigma_{T_1}$ ) is  $15\mu s$ . The star shows the iteration that yielded the best  $T_1$  (see (d) for the decay curve). (f) Histogram of the  $T_1$  values in (b), with a Gaussian fit.

From Eqs. (1) and (2), we find that  $\sigma_{T_1} \propto \mu_{T_1}^{3/2}$  and the data is in agreement with this relationship (dashed line in Fig. 3(c)). Based on this plot, we conclude that Nb qubits capped with Ta and Nb qubits capped with Au exhibit the smallest number of TLS and the highest  $\mu_{T_1}$ , maximum  $T_1$  and fluctuations  $\sigma_{T_1}$ . Conversely, the baseline Nb qubit exhibits the greatest number of TLS and therefore the lowest average  $T_1$ . Assuming TLS loss tangent,  $\delta_{TLS}$ , is proportional to the number of TLS<sup>9</sup>, we plot the relative change of  $\delta_{TLS}$  for each set of devices, as shown in Fig. 3(c). We find 5-8× reduction in loss between the Ta-capped Nb devices and the baseline Nb devices. Additionally, we find that the loss associated with Al-capped Nb qubits is slightly lower than TiN-capped Nb qubits. These results are also in agreement with the qubit  $T_1$  measurements.

To assess the reproducibility, scalability and applicability of this capping method to other substrates, Ta-capped and Nb-only devices (with the same test geometry) were fabricated in the Rigetti Computing quantum integrated circuit foundry (Fremont, California) on high-resistivity Si(100) substrates. More details of this fabrication process can be found in Ref [6]. The Ta-capped devices exhibit a systematic improvement in median  $T_1$  ( $\sim$ 200  $\mu$ s) compared to the baseline Nb qubits ( $\sim$ 120  $\mu$ s). A box plot providing information on the measure-

ments performed on both sets of qubits is provided in Fig. 3b with a maximum measured  $T_1$  value of 451  $\mu s$ . The consistent improvement in median  $T_1$  for the capped Nb devices for both silicon and sapphire substrates supports a performance limitation imposed by the amorphous Nb<sub>2</sub>O<sub>5</sub> surface oxide in un-capped devices. Meanwhile, the  $T_2$ ,  $T_2$  echo, and  $T_\phi$  values are provided in Fig. S2.

Given these promising  $T_1$  results, we also fabricated larger footprint qubit geometries of the Ta-capped Nb (Geometry B and C) on sapphire that are comparable to those used by groups that have demonstrated the largest  $T_1$  values to date  $^{13}$ . In Figure 4, a comparison of the measured T<sub>1</sub> values for a Tacapped device is provided for the 3 different geometries that were investigated. These results are also summarized in Table S2. Geometry B yields the T<sub>1</sub> values with median qubit lifetimes above 300 microseconds, which are in line with the median qubit lifetimes reported by these groups. Further, we observe several individual measurements in excess of 550  $\mu$ s that represent the highest lifetimes reported to date. These results, combined with the fact that this approach is reliable with both silicon and sapphire substrates and can be performed at room temperature makes surface encapsulation a very attractive methodology for achieving high coherence qubits. More-

![](_page_5_Figure_1.jpeg)

<span id="page-5-0"></span>FIG. 4. Effect of Qubit Geometry (a) T<sup>1</sup> comparison of Nb/Ta qubits as a function of three different qubit geometries prepared on sapphire substrates. Geometry B yields the largest T<sup>1</sup> values. (b) Statistics for T<sup>1</sup> consecutively measured over 70 hours for a Geometry B qubit. The average T<sup>1</sup> (µ*T*<sup>1</sup> ) is 323µ*s*. (c) Decay curve associated with measured T<sup>1</sup> value of 586 µ*s* (indicated by star in (b).

over, future studies will involve gold encapsulation and other similar low-loss capping layers with similar larger footprint geometries to potentially push the envelop of performance even further.

Together, these results clearly demonstrate that eliminating Nb2O<sup>5</sup> enhances the T<sup>1</sup> relaxation time of Nb transmon qubits. In the case of devices capped with Ta, Al, and TiN, we still observe the presence of amorphous oxides of a few nm thickness such as Ta2O5, AlO*x*, and TiO*x*, respectively at the sample surface (Fig. [5](#page-6-2) and Fig. [S3\)](#page-4-0). By linking modifications solely to the metal/air interface to measured T<sup>1</sup> values, we observe a trend where Ta2O<sup>5</sup> ranks as the least lossy oxide of those measured, followed by Al2O3, TiO2, and finally, Nb2O5. Further, the loss introduced by these various surface oxides does not appear to be directly correlated to their individual thicknesses as illustrated in Table [II.](#page-5-1) In particular, 1- 2nm TiO*<sup>x</sup>* at the surface of the TiN capped Nb qubit is found to be the thinnest oxide whereas the 5-7 nm Ta2O<sup>5</sup> observed at the surface of the Ta capped Nb qubit prepared on silicon is found to be the thickest.

Therefore, our findings help explain previous experimental studies with qubits prepared from Ta metal exhibit improved T<sup>1</sup> values[13,](#page-7-6)[14](#page-7-7). Namely, our results suggest the improved T<sup>1</sup> predominantly arises from the presence of a less lossy surface oxide, as opposed to the tantalum film being less lossy compared to niobium.

| Substrate Film |    | Surface       | Surface Oxide |
|----------------|----|---------------|---------------|
|                |    | Encapsulation |               |
| Sapphire       | Nb | -             | 3-5nm Nb2O5   |
| Sapphire       | Nb | Ta            | 3-5nm Ta2O5   |
| Sapphire       | Nb | Al            | 2-4nm AlOx    |
| Sapphire       | Nb | TiN           | 1-2nm TiOx    |
| Sapphire       | Nb | Au            | Not Observed  |
| Silicon        | Nb | -             | 3-5nm Nb2O5   |
| Silicon        | Nb | Ta            | 5-7nm Ta2O5   |

<span id="page-5-1"></span>TABLE II. Surface oxides observed on fabricated transmon qubits.

To understand the difference in performance between the capped and baseline samples, we use electron microscopy and, in particular, electron energy loss spectroscopy (EELS) to further evaluate the chemical nature of the Ta oxide. EELS signal captured from points 1-10 on the dark field image of the Ta2O5/Ta interface of the prepared qubit are provided in Fig. [6.](#page-7-17) From this image, we observe that the oxide thickness of Ta is roughly similar to that observed for Nb oxide (4-5 nm). The Ta oxide is found to be predominantly amorphous based on the presence of diffuse diffraction patterns taken in this region (Fig. [S4\)](#page-5-0). Features associated with the tantalum O2,<sup>3</sup> edge are labeled with a dotted line in Fig. [6b](#page-7-17) and those associated with the oxygen K edge are labeled with a dotted line in Fig. [6c](#page-7-17). We find there are changes in the shape of both set of features, but the position of oxygen K remains constant. This indicates that the Ta predominately exists in a 5+ state. This is in contrast to what has been observed in Nb. For Nb, shifts in the features accompanying the onset of the oxygen K edge are observed as a function of position in the oxide due to changes in the valence state[38](#page-8-11)[,39](#page-8-12). This suggests that the Ta oxide present in these capped samples is largely free of substoichiometric regions.

Finally, the x-ray reflectivity signal captured from both the Nb sample as well as the Ta-capped Nb sample is provided in Fig. [S5.](#page-6-2) Based on the data fits performed using dynamical scattering theory[40](#page-8-13), we observe that the surface oxide of Nb consists of roughly 4.1 nm of Nb2O<sup>5</sup> and 0.5 nm of NbO whereas the surface oxide of Ta consists entirely of 5.9 nm of Ta2O5. This technique suggests that the capped samples are largely free of sub-oxides and is consistent with the TEM findings.

Theoretical and experimental findings have linked the variable oxygen content in the Nb oxide layer to the presence of local paramagnetic moments. These moments are a source of flux noise, dephasing, and energy loss[41](#page-8-14)[,42](#page-8-15). We hypothesize that the stoichiometric, predominantly Ta2O<sup>5</sup> layer of Tacapped Nb reduces the potential for moment formation com-

![](_page_6_Figure_1.jpeg)

<span id="page-6-2"></span>FIG. 5. Electron microscopy images of surface oxides observed for (a) Al capped Nb, (b) TiN capped Nb, (c) baseline Nb, and (d) Ta capped Nb qubits prepared on sapphire. Similar images of surface oxides for (e) baseline Nb, and (f) Ta capped Nb qubits prepared on silicon are also presented. The oxides are identified with white arrows and tabulated in Table [II.](#page-5-1)

pared to the variable oxygen content observed in the Nb oxide layer of uncapped Nb. Additionally, it is possible the Ta oxide layer hosts fewer TLS in the frequency range of the qubit frequency compared to the Nb oxide layer. This is an area of continuing active exploration through experimental and theoretical studies. In summary, we have implemented different passivation strategies to eliminate and prevent the formation of lossy Nb surface oxide in Nb superconducting qubits. By capping Nb films with Ta, Al, TiN, and Au, we are able to systematically improve the average qubit relaxation times. Of these capping strategies, we find that the Nb film capped with Ta and Nb film capped with Au yield the highest average T1. We observe similar improvements in average T<sup>1</sup> when superconducting qubits are prepared on silicon as well as sapphire. Together, this methodology offers a solution to delivering state-of-the-art devices with median T<sup>1</sup> times exceeding 300 µs that is compatible with industrial-level processes. Further, this method offers a pathway for continuing to suppress the dielectric loss associated with the metal/air interface through further exploration of capping Nb film with ambient-stable layers prior to air exposure and is applicable to the field of superconducting devices broadly (quantum information science, detectors for cosmic science, and particle accelerators). Finally, this study provides definitive proof that niobium oxides exhibit a larger TLS density than tantalum oxide and will help guide future investigations aimed at building a microscopic understanding of TLS sources in superconducting qubits.

# SUPPLEMENTARY INFORMATION

See supplementary information for wiring diagrams, pulse schemes associated with device measurements, chemical maps, depth profiles and electron diffraction patterns taken from the samples.

## ACKNOWLEDGMENTS

This material is based upon work supported by the U.S. Department of Energy, Office of Science, National Quantum Information Science Research Centers, Superconducting Quantum Materials and Systems Center (SQMS) under contract no. DE-AC02-07CH11359. This work made use of the Pritzker Nanofabrication Facility of the Institute for Molecular Engineering at the University of Chicago, which receives support from Soft and Hybrid Nanotechnology Experimental (SHyNE) Resource (NSF ECCS-2025633), a node of the National Science Foundation's National Nanotechnology Coordinated Infrastructure. The authors thank members of the SQMS Center for valuable discussions.

# DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from the corresponding author upon reasonable request.

# DISCLAIMER

Certain commercial equipment, instruments, or materials are identified in this paper in order to specify the experimental procedure adequately. Such identification is not intended to imply recommendation or endorsement by NIST, nor is it intended to imply that the materials or equipment identified are necessarily the best available for the purpose.

<span id="page-6-0"></span><sup>1</sup>M. Kjaergaard, M. E. Schwartz, J. Braumüller, P. Krantz, J. I.-J. Wang, S. Gustavsson, and W. D. Oliver, [Annual Review of Condensed Mat](https://doi.org/10.1146/annurev-conmatphys-031119-050605)ter Physics 11[, 369 \(2020\),](https://doi.org/10.1146/annurev-conmatphys-031119-050605) [https://doi.org/10.1146/annurev-conmatphys-](https://arxiv.org/abs/https://doi.org/10.1146/annurev-conmatphys-031119-050605)[031119-050605.](https://arxiv.org/abs/https://doi.org/10.1146/annurev-conmatphys-031119-050605)

<sup>2</sup>G. Wendin, [Reports on Progress in Physics](https://doi.org/10.1088/1361-6633/aa7e1a) 80, 106001 (2017).

<span id="page-6-1"></span><sup>3</sup>N. P. de Leon, K. M. Itoh, D. Kim, K. K. Mehta, T. E. Northup, H. Paik, B. S. Palmer, N. Samarth, S. Sangtawesin, and D. W. Steuerman, Science 372, [10.1126/science.abb2823](https://doi.org/10.1126/science.abb2823) (2021).

![](_page_7_Figure_1.jpeg)

<span id="page-7-17"></span>FIG. 6. (a) Dark field STEM image of the Ta2O5/Ta interface. Electron energy loss spectra (EELS) taken from the locations indicated in (a) are provided in (b) and (c). (b) EELS signal taken from the specified region demonstrating how the tantalum O2,<sup>3</sup> edge evolves with position. (c) EELS signal taken from the specified region demonstrating how oxygen K edge evolves as a function of position. The dotted lines indicate minimal variation in the position of various features of the spectra. This suggests that the oxygen stoichiometry remains consistent throughout the oxide region.

<span id="page-7-0"></span>4 J. J. Burnett, A. Bengtsson, M. Scigliuzzo, D. Niepce, M. Kudra, P. Delsing, and J. Bylander, [npj Quantum Information](https://doi.org/10.1038/s41534-019-0168-5) 5, 54 (2019).

<sup>5</sup>S. Schlör, J. Lisenfeld, C. Müller, A. Bilmes, A. Schneider, D. P. Pappas, A. V. Ustinov, and M. Weides, Phys. Rev. Lett. 123[, 190502 \(2019\).](https://doi.org/10.1103/PhysRevLett.123.190502)

<span id="page-7-1"></span><sup>6</sup>A. Nersisyan, S. Poletto, N. Alidoust, R. Manenti, R. Renzas, C.-V. Bui, K. Vu, T. Whyland, Y. Mohan, E. A. Sete, S. Stanwyck, A. Bestwick, and M. Reagor, in *[2019 IEEE International Electron Devices Meeting \(IEDM\)](https://doi.org/10.1109/IEDM19573.2019.8993458)* (2019) pp. 31.1.1–31.1.4.

<span id="page-7-2"></span><sup>7</sup>C. D. Wilen, S. Abdullah, N. A. Kurinsky, C. Stanford, L. Cardani, G. D'Imperio, C. Tomei, L. Faoro, L. B. Ioffe, C. H. Liu, A. Opremcak, B. G. Christensen, J. L. DuBois, and R. McDermott, [Nature](https://doi.org/10.1038/s41586-021-03557-5) 594, 369 [\(2021\).](https://doi.org/10.1038/s41586-021-03557-5)

<sup>8</sup>R. W. Simmonds, K. Lang, D. Hite, S. Nam, D. P. Pappas, and J. M. Martinis, Phys. Rev. Lett. 93, 077003 (2004).

<span id="page-7-16"></span><sup>9</sup>C. Müller, J. H. Cole, and J. Lisenfeld, [Reports on Progress in Physics](https://doi.org/10.1088/1361-6633/ab3a7e) 82, [124501 \(2019\).](https://doi.org/10.1088/1361-6633/ab3a7e)

<span id="page-7-3"></span><sup>10</sup>R. McDermott, IEEE Transactions on Applied Superconductivity 19, 2 (2009).

<span id="page-7-4"></span><sup>11</sup>S. K. Tolpygo, V. Bolkhovsky, T. J. Weir, L. M. Johnson, M. A. Gouker, and W. D. Oliver, [IEEE Transactions on Applied Superconductivity](https://doi.org/10.1109/TASC.2014.2374836) 25, 1 [\(2015\).](https://doi.org/10.1109/TASC.2014.2374836)

<span id="page-7-5"></span><sup>12</sup>A. Romanenko, R. Pilipenko, S. Zorzetti, D. Frolov, M. Awida, S. Belomestnykh, S. Posen, and A. Grassellino, [Phys. Rev. Applied](https://doi.org/10.1103/PhysRevApplied.13.034032) 13, 034032 [\(2020\).](https://doi.org/10.1103/PhysRevApplied.13.034032)

<span id="page-7-6"></span><sup>13</sup>A. P. M. Place, L. V. H. Rodgers, P. Mundada, B. M. Smitham, M. Fitzpatrick, Z. Leng, A. Premkumar, J. Bryon, A. Vrajitoarea, S. Sussman, G. Cheng, T. Madhavan, H. K. Babla, X. H. Le, Y. Gang, B. Jäck, A. Gyenis, N. Yao, R. J. Cava, N. P. de Leon, and A. A. Houck, [Nature Communi](https://doi.org/10.1038/s41467-021-22030-5)cations 12[, 1779 \(2021\).](https://doi.org/10.1038/s41467-021-22030-5)

<span id="page-7-7"></span><sup>14</sup>C. Wang, X. Li, H. Xu, Z. Li, J. Wang, Z. Yang, Z. Mi, X. Liang, T. Su, C. Yang, G. Wang, W. Wang, Y. Li, M. Chen, C. Li, K. Linghu, J. Han, Y. Zhang, Y. Feng, Y. Song, T. Ma, J. Zhang, R. Wang, P. Zhao, W. Liu, G. Xue, Y. Jin, and H. Yu, [npj Quantum Information](https://doi.org/10.1038/s41534-021-00510-2) 8, 3 (2022).

<span id="page-7-8"></span><sup>15</sup>A. Romanenko and D. I. Schuster, Phys. Rev. Lett. 119[, 264801 \(2017\).](https://doi.org/10.1103/PhysRevLett.119.264801)

<span id="page-7-15"></span><sup>16</sup>A. A. Murthy, P. Masih Das, S. M. Ribet, C. Kopas, J. Lee, M. J. Reagor, L. Zhou, M. J. Kramer, M. C. Hersam, M. Checchin, A. Grassellino, R. d. Reis, V. P. Dravid, and A. Romanenko, ACS Nano 16[, 17257 \(2022\).](https://doi.org/10.1021/acsnano.2c07913)

<sup>17</sup>D. Niepce, J. J. Burnett, M. G. Latorre, and J. Bylander, [Superconductor](https://doi.org/10.1088/1361-6668/ab6179) [Science and Technology](https://doi.org/10.1088/1361-6668/ab6179) 33, 025013 (2020).

<sup>18</sup>D. Bafia, A. Grassellino, and A. Romanenko, [Probing the role of low tem](https://doi.org/10.48550/ARXIV.2108.13352)[perature vacuum baking on photon lifetimes in superconducting niobium](https://doi.org/10.48550/ARXIV.2108.13352) [3-d resonators](https://doi.org/10.48550/ARXIV.2108.13352) (2021).

<sup>19</sup>J. Burnett, L. Faoro, and T. Lindström, [Superconductor Science and Tech](https://doi.org/10.1088/0953-2048/29/4/044008)nology 29[, 044008 \(2016\).](https://doi.org/10.1088/0953-2048/29/4/044008)

<span id="page-7-9"></span><sup>20</sup>A. Premkumar, C. Weiland, S. Hwang, B. Jäck, A. P. M. Place, I. Waluyo, A. Hunt, V. Bisogni, J. Pelliciari, A. Barbour, M. S. Miller, P. Russo, F. Camino, K. Kisslinger, X. Tong, M. S. Hybertsen, A. A. Houck, and I. Jarrige, [Communications Materials](https://doi.org/10.1038/s43246-021-00174-7) 2, 72 (2021).

<span id="page-7-10"></span><sup>21</sup>C. R. H. McRae, H. Wang, J. Gao, M. R. Vissers, T. Brecht, A. Dunsworth, D. P. Pappas, and J. Mutus, [Review of Scientific Instruments](https://doi.org/10.1063/5.0017378) 91, 091101 [\(2020\),](https://doi.org/10.1063/5.0017378) [https://doi.org/10.1063/5.0017378.](https://arxiv.org/abs/https://doi.org/10.1063/5.0017378)

<span id="page-7-11"></span><sup>22</sup>M. Checchin, D. Frolov, A. Lunin, A. Grassellino, and A. Romanenko, [Phys. Rev. Appl.](https://doi.org/10.1103/PhysRevApplied.18.034013) 18, 034013 (2022).

<span id="page-7-12"></span><sup>23</sup>J. Verjauw, A. Potocnik, M. Mongillo, R. Acharya, F. Mohiyaddin, ˇ G. Simion, A. Pacco, T. Ivanov, D. Wan, A. Vanleenhove, L. Souriau, J. Jussot, A. Thiam, J. Swerts, X. Piao, S. Couet, M. Heyns, B. Govoreanu, and I. Radu, [Phys. Rev. Applied](https://doi.org/10.1103/PhysRevApplied.16.014018) 16, 014018 (2021).

<span id="page-7-13"></span><sup>24</sup>M. V. P. Altoé, A. Banerjee, C. Berk, A. Hajr, A. Schwartzberg, C. Song, M. Alghadeer, S. Aloni, M. J. Elowson, J. M. Kreikebaum, E. K. Wong, S. M. Griffin, S. Rao, A. Weber-Bargioni, A. M. Minor, D. I. Santiago, S. Cabrini, I. Siddiqi, and D. F. Ogletree, PRX Quantum 3[, 020312 \(2022\).](https://doi.org/10.1103/PRXQuantum.3.020312)

<span id="page-7-14"></span><sup>25</sup>J. Knobloch, [AIP Conference Proceedings](https://doi.org/10.1063/1.1597364) 671, 133 (2003), [https://aip.scitation.org/doi/pdf/10.1063/1.1597364.](https://arxiv.org/abs/https://aip.scitation.org/doi/pdf/10.1063/1.1597364)

<sup>26</sup>A. Romanenko, F. Barkov, L. D. Cooley, and A. Grassellino, [Superconduc](https://doi.org/10.1088/0953-2048/26/3/035003)[tor Science and Technology](https://doi.org/10.1088/0953-2048/26/3/035003) 26, 035003 (2013).

- <span id="page-8-0"></span><sup>27</sup>J. Lee, Z. Sung, A. A. Murthy, M. Reagor, A. Grassellino, and A. Romanenko, Discovery of nb hydride precipitates in superconducting qubits (2021), [arXiv:2108.10385 \[quant-ph\].](https://arxiv.org/abs/2108.10385)
- <span id="page-8-1"></span><sup>28</sup>K. Zheng, D. Kowsari, N. J. Thobaben, X. Du, X. Song, S. Ran, E. A. Henriksen, D. S. Wisbey, and K. W. Murch, Applied Physics Letters 120 , [10.1063/5.0082755](https://doi.org/10.1063/5.0082755) (2022), 102601, [https://pubs.aip.org/aip/apl/article](https://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/doi/10.1063/5.0082755/16445280/102601_1_online.pdf)[pdf/doi/10.1063/5.0082755/16445280/102601\\_1\\_online.pdf.](https://arxiv.org/abs/https://pubs.aip.org/aip/apl/article-pdf/doi/10.1063/5.0082755/16445280/102601_1_online.pdf)
- <span id="page-8-2"></span><sup>29</sup>X. Fang, J.-S. Oh, M. Kramer, A. Romanenko, A. Grassellino, J. Zasadzinski, and L. Zhou, [Materials Research Letters](https://doi.org/10.1080/21663831.2022.2126737) 11, 108 (2023), [https://doi.org/10.1080/21663831.2022.2126737.](https://arxiv.org/abs/https://doi.org/10.1080/21663831.2022.2126737)
- <span id="page-8-3"></span><sup>30</sup>A. A. Murthy, J. Lee, C. Kopas, M. J. Reagor, A. P. McFadden, D. P. Pappas, M. Checchin, A. Grassellino, and A. Romanenko, [App. Phys. Lett.](https://doi.org/10.1063/5.0079321) 120 , [044002 \(2022\),](https://doi.org/10.1063/5.0079321) [https://doi.org/10.1063/5.0079321.](https://arxiv.org/abs/https://doi.org/10.1063/5.0079321)
- <span id="page-8-4"></span><sup>31</sup>X. Lu, D. P. Goronzy, C. G. Torres-Castanedo, P. Masih Das, M. Kazemzadeh-Atoufi, A. McFadden, C. R. H. McRae, P. W. Voorhees, V. P. Dravid, M. J. Bedzyk, M. C. Hersam, and J. M. Rondinelli, [Phys. Rev.](https://doi.org/10.1103/PhysRevMaterials.6.064402) Mater. 6[, 064402 \(2022\).](https://doi.org/10.1103/PhysRevMaterials.6.064402)
- <span id="page-8-10"></span><sup>32</sup>X. You, Z. Huang, U. Alyanak, A. Romanenko, A. Grassellino, and S. Zhu, Phys. Rev. App. 18, 044026 (2022).
- <span id="page-8-5"></span><sup>33</sup>A. A. Clerk and D. W. Utami, Phys. Rev. A 75[, 042302 \(2007\).](https://doi.org/10.1103/PhysRevA.75.042302)
- <span id="page-8-6"></span><sup>34</sup>A. Vaaranta, M. Cattaneo, and R. E. Lake, [Phys. Rev. A](https://doi.org/10.1103/PhysRevA.106.042605) 106, 042605 [\(2022\).](https://doi.org/10.1103/PhysRevA.106.042605)
- <span id="page-8-7"></span><sup>35</sup>P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, App. Phys. Rev. 6, 021318 (2019).
- <span id="page-8-8"></span><sup>36</sup>J. J. Burnett, A. Bengtsson, M. Scigliuzzo, D. Niepce, M. Kudra, P. Delsing, and J. Bylander, npj Quantum Information 5, 54 (2019).
- <span id="page-8-9"></span><sup>37</sup>P. Klimov, J. Kelly, Z. Chen, M. Neeley, A. Megrant, B. Burkett, R. Barends, K. Arya, B. Chiaro, Y. Chen, *et al.*, Phys. Rev. Lett. 121 , 090502 (2018).
- <span id="page-8-11"></span><sup>38</sup>R. Tao, R. Todorovic, J. Liu, R. J. Meyer, A. Arnold, W. Walkosz, P. Zapol, A. Romanenko, L. D. Cooley, and R. F. Klie, [Journal of Applied Physics](https://doi.org/10.1063/1.3665193) 110[, 124313 \(2011\),](https://doi.org/10.1063/1.3665193) [https://doi.org/10.1063/1.3665193.](https://arxiv.org/abs/https://doi.org/10.1063/1.3665193)
- <span id="page-8-12"></span><sup>39</sup>J.-S. Oh, X. Fang, T.-H. Kim, M. Lynn, M. Kramer, M. Zarea, J. A. Sauls, A. Romanenko, S. Posen, A. Grassellino, C. J. Kopas, M. Field, J. Marshall, H. Cansizoglu, J. Y. Mutus, M. Reagor, and L. Zhou, [Multi-modal electron](https://doi.org/10.48550/ARXIV.2204.06041) [microscopy study on decoherence sources and their stability in nb based](https://doi.org/10.48550/ARXIV.2204.06041) [superconducting qubit](https://doi.org/10.48550/ARXIV.2204.06041) (2022).
- <span id="page-8-13"></span><sup>40</sup>A. Nelson, [Journal of Applied Crystallography](https://doi.org/https://doi.org/10.1107/S0021889806005073) 39, 273 (2006), [https://onlinelibrary.wiley.com/doi/pdf/10.1107/S0021889806005073.](https://arxiv.org/abs/https://onlinelibrary.wiley.com/doi/pdf/10.1107/S0021889806005073)
- <span id="page-8-14"></span><sup>41</sup>E. Sheridan, T. F. Harrelson, E. Sivonxay, K. A. Persson, M. V. P. Altoé, I. Siddiqi, D. F. Ogletree, D. I. Santiago, and S. M. Griffin, Microscopic theory of magnetic disorder-induced decoherence in superconducting nb films (2021), arxiv.2111.11684, arXiv, https://arxiv.org/abs/2111.11684, accessed February 26, 2023.
- <span id="page-8-15"></span><sup>42</sup>T. Proslier, M. Kharitonov, M. Pellin, J. Zasadzinski, and Ciovati, [IEEE](https://doi.org/10.1109/TASC.2011.2107491) [Trans. Appl. Supercond.](https://doi.org/10.1109/TASC.2011.2107491) 21, 2619 (2011).
- <span id="page-8-16"></span><sup>43</sup>M. R. Vissers, J. Gao, D. S. Wisbey, D. A. Hite, C. C. Tsuei, A. D. Corcoles, M. Steffen, and D. P. Pappas, [Applied Physics Letters](https://doi.org/10.1063/1.3517252) 97, 232509 (2010), [https://doi.org/10.1063/1.3517252.](https://arxiv.org/abs/https://doi.org/10.1063/1.3517252)
- <span id="page-8-17"></span><sup>44</sup>G. J. Dolan, [Applied Physics Letters](https://doi.org/10.1063/1.89690) 31, 337 (1977), [https://doi.org/10.1063/1.89690.](https://arxiv.org/abs/https://doi.org/10.1063/1.89690)
- <span id="page-8-18"></span><sup>45</sup>R. Gordon, C. Murray, C. Kurter, M. Sandberg, S. Hall, K. Balakrishnan, R. Shelby, B. Wacaser, A. Stabile, J. Sleight, *et al.*, Applied Physics Letters 120, 074002 (2022).
- <span id="page-8-19"></span><sup>46</sup>A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, Reviews of Modern Physics 93, 025005 (2021).
- <span id="page-8-20"></span><sup>47</sup>S. Krinner, S. Storz, P. Kurpiers, P. Magnard, J. Heinsoo, R. Keller, J. Luetolf, C. Eichler, and A. Wallraff, EPJ Quantum Technology 6, 2 (2019).

#### SUPPLEMENTARY INFORMATION

#### **METHODS**

## **Qubit Fabrication**

The first four sets of qubits were fabricated on 550  $\mu$ m thick c-plane sapphire wafers (double-side polished, HEMEX grade HEM Sapphire) at the Pritzker Nanofabrication (PNF) facility. These wafers were first solvent cleaned by sonicating in n-methyl-2-pyrrolidone (NMP) heated to 80°C followed by sequential soaks in isopropanol, acetone, isopropanol, and deionized (DI) water under mild sonication at room temperature. The sapphire wafers were subject to further cleaning in a mixture of DI water, 30% ammonium hydroxide in water, 30% hydrogen peroxide in water with a 5:1:1 ratio. This cleaning was performed at 65°C for 5 minutes followed by DI water rinse. Following the surface treatments, the substrate was immediately placed inside an AJA ATC 2200 sputtering system with a base pressure less than  $10^{-7}$  Torr. After dehydrating the substrate under vacuum at 200°C for 30 minutes, the wafers were allowed to cool-down before Nb was sputtered at room temperature. DC magnetron sputtering was performed using a 3-inch diameter Nb target with a metals basis purity of 99.95% with an Ar flow rate of 30 sccm and partial pressure of 3.5 mTorr. A sputtering power of 600 W was used and the substrate was rotated at 20 rpm. These conditions resulted in a deposition rate of approximately 19 nm/min. For the qubits prepared on silicon substrates the process involved first cleaning the substrate with a RCA surface treatment detailed previously<sup>43</sup> followed by a buffered oxide etch, before immediately loading into the sputter deposition system<sup>6</sup>. In this case, Nb films were deposited by high power impulse magnetron sputtering with a 6-inch diameter Nb target having a metal basis purity of 99.9995% with a base pressure less than 1E-8 Torr at room temperature.

For the Nb samples capped with Ta or Al, the capping layers were immediately sputtered onto the Nb film *in situ*, i.e. without breaking vacuum in the deposition chamber. Al was deposited from a 3-inch diameter Al target (metals basis purity of 99.9995%) with a DC sputtering power of 500 W. Ta was deposited from a 3-inch diameter Ta target (metals basis purity of 99.95%) with an RF sputtering power of 600 W. The flow rate and substrate rotation speed were kept constant. The Ar partial pressure was increased to 5 mTorr for the Ta encapsulation layer.

For the Nb films capped with TiN, the sample was removed from the sputter deposition system and immediately loaded into an Ultratech/Cambridge Fiji G2 Plasma-Enhanced ALD system for atomic layer deposition of TiN. The TiN ALD recipe comprised of a 250 ms TDMAT, (Ti(N(CH<sub>3</sub>)<sub>2</sub>)<sub>4</sub>) precursor dose and a 30 s N<sub>2</sub> plasma exposure. Each of these pulses was separated by a 5 s purge. This process was repeated for 40 cycles and the sample was held at 270°C during the course of this deposition. These capped layers were all between 10-15 nm in thickness.

For the Nb films capped with Au, the sample was removed from the sputter deposition system and immediately loaded into an Angstrom EvoVac Electron Beam Evaporator system with a base pressure of 2E-7 Torr. The Nb surface oxide was milled away with Ar<sup>+</sup> ions for 15 min with a beam voltage of 400 V, an accelerator voltage of 80 V and an ion current of 15 mA. A 10 nm layer of Au was subsequently deposited using e-beam deposition at a rate of 1 angstrom/second. During this process, the substrate was rotated at 10 rpm.

Following film deposition, the superconducting microwave circuit was patterned with photolithography using a Heidelberg MLA 150 Advanced Maskless Aligner. After exposure and development of the photoresist, the circuit layer was defined by dry etching using an inductively coupled plasma - reactive ion etching (ICP-RIE) system. Dry etching was performed using a mixture of  $Cl_2:BCl_3:Ar$  with flow rates 30:10:10 sccm. The etching process was halted following the etching of the Nb metal as well as some of the underlying substrate. In the case of sapphire, the substrate was overetched by 20nm and in the case of silicon, the substrate was overetched by 70nm. The remaining photoresist was stripped from the patterned substrate first by ashing in  $O_2$  (to remove the hardened photoresist on the surface) and then by soaking in NMP heated to  $80^{\circ}$ C for several hours.

The Dolan bridge style Al/AlOx/Al Josephson junctions are fabricated using the well-known double angle shadow evaporation technique and solvent lift off for the samples prepared on sapphire<sup>44</sup>. The junction patterns are exposed using an electron beam with an accelerating voltage of 100kV. The JJ layers for the samples prepared on silicon are deposited using a double-angle bridge-less lift-off lithography process. In a subsequent lithography and metal lift-off step, galvanic connection between the Josephson junctions and the microwave circuitry is established. Ar ion milling is employed to remove surface oxides immediately prior to Al deposition. Following lift-off of the junction leads, the surface is exposed to a mild UV ozone treatment for 10 min to remove residual resist.

The geometry of the qubit consists of a pair of rectangular shunting capacitor paddles joined by a single Josephson junction as seen in Fig. 1a. The paddle size, hence the capacitance, is varied while the gap is kept fixed. A geometry similar to this has exhibited greater than 300 µs qubit coherence times in the past<sup>45</sup>. To minimize the influence of the magnetic flux noise, all qubits are designed to have fixed frequencies. Following the circuit quantum electrodynamics (cOED) architecture<sup>46</sup>, each qubit is capacitively coupled to a single quarter-wave readout resonator. Each of these resonators is inductively coupled to a 50  $\Omega$  transmission line where a microwave probe tone can be sent to the resonator. On a single  $7.5 \times 7.5$  mm<sup>2</sup> sapphire chip, there are four qubits with this rectangular-paddle geometry. The resonator frequencies are evenly spaced between 6.8 and 7.2 GHz with qubit frequencies ranging from 3.4-6 GHz as summarized in Tables S1 and

## A. Qubit Measurement Setup and Method

Each qubit chip was packaged in a gold plated copper box, where it is mounted on a cold finger directly attached to the

dilution refrigerator (DR) mixing plate. Both the enclosure and the cold finger are made of gold-coated copper to reduce thermal resistance between the qubit chip and the enclosure. Each qubit chip is protected from IR photons with Eccosorb filters on both input and output lines. All qubit enclosures and Eccosorb filters are enclosed within magnetic shielding to reduce flux noise and are wired to cryogenic switches anchored on the mixing stage. This way, all qubits share the same microwave input and output lines, which enables comparison of different qubits. A total of 52 dB attenuation is distributed across different temperature stages to effectively suppress thermal noise below 10−<sup>3</sup> noise photon number[47](#page-8-20) . Three cryogenic isolators with a total of 60 dB isolation at the output line reduces the backaction noise arising from the high electron mobility transistor (HEMT) amplifier mounted on the 3K stage. Low-pass filters (K&L Microwave) are used to remove the high-frequency noise above 10 GHz.

The qubit state is measured via dispersive readout, in which the qubit frequency is far detuned from the resonator frequency (∆ = *f<sup>q</sup>* − *fr*) so that ∆ ≫ κ,*g*, where κ and *g* are the resonator linewidth and qubit-resonator coupling strengths, respectively[35](#page-8-7). In this case, the qubit and resonator do not directly exchange energy. Instead, the qubit induces a statedependent frequency shift of the resonator. By interrogating the resonator, the qubit state can be inferred.

Qubit measurements of devices on Si substrates were performed in a different dilution refrigerator with a similar configuration. Chips were attached to a thermally anchored printed circuit board on the mixing chamber plate of a dilution refrigerator with a base temperature of 10 mK. The devices are shielded from magnetic fields with superconducting and cryoperm cans with a blackbody absorber to suppress stray infrared radiation. The input signal chain has a total attenuation of 76 dB achieved by a series of attenuators anchored at different temperatures, and a 7.65 GHz low-pass filter. The output signal line is filtered by isolators and amplified by a HEMT at 4K and a series of room-temperature amplifiers. Relaxation times are measured in the dispersive regime continuously on each device over multiple days.

# 1. TEM Sample Preparation

TEM samples were prepared using a 30 kV focused Ga<sup>+</sup> ion beam. In order to protect the surface oxide during the ion milling process, the sample was first coated with 50 nm of carbon. The samples were finely polished to a thickness of roughly 50 nm using 5 kV and 2 kV Ga<sup>+</sup> ions in an effort to remove surface damage and amorphization in the regions of interest.

# 2. STEM Data Collection

EDS data was acquired using a JEOL ARM 200CF ARM S/TEM operated at 200 kV. The camera length is set to 8 cm and the condenser aperture is selected to provide a convergence semi-angle of 30 mrad with beam current ∼ 20 pA in order to minimize beam-induced damage. The intensity of the Nb Lα and Ta Lα signals were plotted as a function of position for the Ta-capped Nb sample. The intensity of the Nb Lα and Al Kα signals were plotted as a function of position for the Al-capped Nb sample. The intensity of the Nb Lα, Ti Kα, and N Kα signals were plotted as a function of position for the TiN-capped Nb sample. The intensity of the Nb Kα and Au Lα signals were plotted as a function of position for the Au-capped Nb sample. In order to prepare the chemical maps presented in Fig. 2a-c, the individual chemical maps for each sample were normalized, overlaid, and lowpass filtered using a Gaussian function with σ = 2. The EELS data and associated HAADF STEM image was acquired using a Titan Themis with GIF quantum ER system. Electron diffraction patterns from the Ta oxide was collected on JEOL 300F Grand ARM S/TEM using an accelerating voltage of 300kV. In this case, the camera length is set to 20 cm and the condenser aperture is selected to provide a convergence semiangle of 1 mrad. Four-dimensional STEM (4D-STEM) data sets[16](#page-7-15) were acquired in a 200 x 100 mesh (with pixel size of 1.6 nm) across the thin films using a Gatan OneView camera and synchronized using STEMx.

# B. Qubit T1 Measurements

| Geometry | Film   | Qubit Frequency (MHz) Average T1 (µs) Q (millions) |     |     |
|----------|--------|----------------------------------------------------|-----|-----|
| A        | Nb     | 5986                                               | 54  | 2.0 |
| A        | Nb     | 5796                                               | 21  | 0.8 |
| A        | Nb/TiN | 4990                                               | 60  | 1.9 |
| A        | Nb/TiN | 4721                                               | 105 | 3.1 |
| A        | Nb/TiN | 4645                                               | 55  | 1.6 |
| A        | Nb/Al  | 4894                                               | 77  | 2.4 |
| A        | Nb/Al  | 4968                                               | 69  | 2.2 |
| A        | Nb/Al  | 4586                                               | 86  | 2.5 |
| A        | Nb/Al  | 4512                                               | 78  | 2.2 |
| A        | Nb/Au  | 4637                                               | 80  | 2.3 |
| A        | Nb/Au  | 4077                                               | 134 | 3.4 |
| A        | Nb/Au  | 4185                                               | 100 | 2.6 |
| A        | Nb/Au  | 4193                                               | 120 | 3.2 |
| A        | Nb/Au  | 4657                                               | 109 | 3.2 |
| A        | Nb/Au  | 4749                                               | 137 | 4.1 |
| A        | Nb/Au  | 4782                                               | 27  | 0.8 |
| A        | Nb/Ta  | 5005                                               | 102 | 3.2 |
| A        | Nb/Ta  | 4746                                               | 118 | 3.5 |
| A        | Nb/Ta  | 4707                                               | 103 | 3.0 |
| A        | Nb/Ta  | 4488                                               | 161 | 4.5 |

TABLE S1. Summary of qubit devices prepared with different capping layers.

| Geometry | Film  | Qubit Frequency (MHz) Average T1 (µs) Q (millions) |     |     |
|----------|-------|----------------------------------------------------|-----|-----|
| A        | Nb/Ta | 3476                                               | 131 | 2.9 |
| A        | Nb/Ta | 3485                                               | 136 | 3.0 |
| A        | Nb/Ta | 3477                                               | 186 | 4.1 |
| A        | Nb/Ta | 3530                                               | 120 | 2.7 |
| B        | Nb/Ta | 3945                                               | 280 | 6.9 |
| B        | Nb/Ta | 3952                                               | 324 | 8.0 |
| B        | Nb/Ta | 3814                                               | 262 | 6.3 |
| B        | Nb/Ta | 3930                                               | 214 | 5.3 |
| C        | Nb/Ta | 3737                                               | 193 | 4.5 |
| C        | Nb/Ta | 3634                                               | 199 | 4.5 |
| C        | Nb/Ta | 3904                                               | 109 | 2.7 |

TABLE S2. Summary of Nb/Ta qubit devices with different geometries.

![](_page_13_Figure_1.jpeg)

FIG. S1. ToF-SIMS Depth profile taken from the surface of Nb film. The Nb2O<sup>5</sup> region is labeled. Fig. [2e](#page-3-0) is constructed by taking the integrated counts of the Nb2O<sup>5</sup> <sup>−</sup> for each of the samples using the same ion beam conditions.

![](_page_14_Figure_1.jpeg)

FIG. S2. Box plot displaying measured (a) T2, (b) T<sup>2</sup> echo, and (c) T<sup>φ</sup> values for test devices fabricated on silicon substrates. Boxes mark the 25th percentile and the 75th percentile of the measurement distribution. The line inside each box represents the median value, and whiskers or diamonds represent outliers. The T2, T<sup>2</sup> echo, and T<sup>φ</sup> values are not heavily impacted by capping Nb with Ta.

![](_page_15_Figure_1.jpeg)

FIG. S3. ToF-SIMS depth profiles taken from the surface of Nb films capped with TiN, Al, and Ta. TiO2, Al2O3, and Ta2O5, respectively are observed at the surface of these films.

![](_page_16_Figure_1.jpeg)

FIG. S4. (a) DF image of Ta oxide/Ta interface using a virtual detector that matches the diffraction ring of the Ta oxide. The dotted arrows indicate regions over which changes in the diffraction pattern are provided in (c-e). (b) Diffraction pattern taken from  $\text{TaO}_x$  with a virtual annular detector. The radius of the inner ring is  $2.07 \text{ nm}^{-1}$  and the radius of the outer ring is  $3.25 \text{ nm}^{-1}$  in order to preferentially produce a  $\text{TaO}_x$  dark field image. (c-f) Changes in the electron diffraction pattern in the direction of the arrows indicated in (a) for the labeled regions 1, 2, 3, and 4, respectively. In all cases, a diffuse diffraction pattern, which is characteristic of an amorphous solid is observed for the  $\text{TaO}_x$ . Intense diffraction spots tend to appear when moving towards the crystalline Ta metal.

![](_page_17_Figure_1.jpeg)

FIG. S5. X-ray reflectivity (a) and x-ray diffraction (b) pattern taken from both the Nb sample and the Nb sample capped with Ta. Fits based on dynamical scattering suggests that the surface oxide of Nb consists of roughly 4.1 nm of Nb2O<sup>5</sup> and 0.5 nm of NbO whereas the surface oxide of Ta consists entirely of 5.9 nm of Ta2O5. X-ray diffraction pattern suggest both sets of films exhibit the body-centered cubic (BCC) crystal structure and predominantly exhibit {110} texture. For Ta, this crystal structure is associated with the α phase.