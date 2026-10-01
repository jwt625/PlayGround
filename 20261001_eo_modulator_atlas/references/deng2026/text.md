---
paper_id: deng2026
source_url: https://doi.org/10.1038/s41377-025-02081-9
doi: 10.1038/s41377-025-02081-9
license: CC-BY-4.0
sha256: f41dcedbfbaedbbb3ba87fb2867f619af2e9c3b7d029f2a0fe7595a1c4b167d4
pages: 10
extracted_on: 2026-10-01
extraction_method: pymupdf get_text('text'); tables and equations may be garbled; plotted curves are not digitized
---

<!-- page 1 -->
Deng et al. Light: Science & Applications (2026) 15:21 
www.nature.com/lsa
https://doi.org/10.1038/s41377-025-02081-9
A R T I C L E
O p e n A c c e s s
Self-buffered epitaxy of barium titanate on oxide
insulators enables high-performance electro-optic
modulators
Chenguang Deng1, Yutong He2, Wenfeng Yang
1, Han Yu1, Zijian Hong
3,4,5, Hao Liu2, Haojie Han1, Wei Li1,
Yunpeng Ma1, Zhongshan Zhang6, Yongjun Wu3,4,5, Jing Ma
1, Bing Xiong
2, Changzheng Sun2✉, Rong Yu
1,
Jing-Feng Li
1, Ji Zhou1, Yi Luo2 and Qian Li
1✉
Abstract
Integrated photonics has emerged as a promising alternative for data communication and computing, ferroelectric
BaTiO3 (BTO) stands out for its exceptional electro-optic response among candidate materials. However, direct epitaxial
growth of BTO entails a fundamental trade-off: substrates with low refractive index are required for strong optical
conﬁnement, yet those with large lattice mismatch degrade ﬁlm crystalline quality and electro-optic performance. We
report a buffer-free, strain-engineered approach to integrate high-performance BTO thin ﬁlms directly on LaAlO3-
Sr2TaAlO6 (LSAT) oxide-insulator substrates. By exploiting a self-buffer layer formed during the initial growth stage, we
achieve periodic in-plane strain modulation that stabilizes a polymorphic phase boundary with orthorhombic polar
nanoregions, yielding a Pockels coefﬁcient exceeding 358 pm V⁻¹ and a Curie temperature raised to 200 °C. Leveraging
this material platform, we demonstrate the ﬁrst realization of a Mach–Zehnder modulator using epitaxial BTO on LSAT.
The device exhibits a half-wave voltage–length product of 0.7 V cm at 1550 nm, which closely matches ﬁnite-element
simulations, and supports a 6-dB electro-optic bandwidth of 28 GHz. Our results validate BTO on LSAT as a viable
photonic platform for scalable, low-voltage and high-speed modulators.
Introduction
As transistor miniaturization approaching fundamental
physical limits, integrated photonics has gained increasing
attention as an emerging alternative in recent years1. In
data communication, integrated photonic platforms offer
inherent advantages such as high bandwidth and low
transmission loss, where high-performance electro-optic
(EO) modulators are essential components2,3. Although
silicon photonic modulators based on the plasmonic
dispersion effect have been developed, their modulation
efﬁciency and speed remain limited4. Thin-ﬁlm LiNbO3
has also attracted considerable interest5,6 yet its produc-
tion relies on complex and expensive ion slicing and
bonding processes7. Moreover, the intrinsic low EO
coefﬁcients of both silicon and LiNbO3 further limit their
suitability for dense photonic integration. In contrast,
ferroelectric
perovskite
BaTiO3
(BTO)
has
recently
emerged as a promising candidate for integrated photo-
nics due to its outstanding EO response8,9. Signiﬁcant
progress has been made in employing BTO thin ﬁlms for
EO
modulation
with
multi-functionality,
including
cryogenic-temperature
modulation10
and
non-volatile
photonic phase shifting11. Although Pockels EO coefﬁ-
cients ranging up to an order of magnitude higher than
that of LiNbO3 have been reported, they remain sig-
niﬁcantly lower than the bulk value12. Direct bonding of
BTO to SiO2/Si wafer can partially restore its intrinsic EO
© The Author(s) 2026
OpenAccessThisarticleislicensedunderaCreativeCommonsAttribution4.0InternationalLicense,whichpermitsuse,sharing,adaptation,distributionandreproduction
in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if
changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If
material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain
permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.
Correspondence: Changzheng Sun (czsun@tsinghua.edu.cn) or
Qian Li (qianli_mse@tsinghua.edu.cn)
1State Key Laboratory of New Ceramic Materials, School of Materials Science
and Engineering, Tsinghua University, Beijing, China
2Beijing National Research Centre for Information Science and Technology
(BNRist), State Key Laboratory of Space Network and Communications,
Department of Electronic Engineering, Tsinghua University, Beijing, China
Full list of author information is available at the end of the article
These authors contributed equally: Chenguang Deng, Yutong He, Wenfeng
Yang, Han Yu
1234567890():,;
1234567890():,;
1234567890():,;
1234567890():,;

<!-- page 2 -->
response13, yet developing simple and cost-effective epi-
taxial growth strategies on oxide insulator platforms
remains essential for realizing high-performance and
scalable BTO-based devices.
Addressing these challenges requires not only suitable
growth methods but also careful selection of low-
refractive-index substrates that support strong optical
conﬁnement. Although SrTiO3 (n = 2.28 at 1550 nm)
substrates allow high-quality BTO ﬁlm growth, its com-
parable refractive index with BTO (n = 2.26 at 1550 nm)
severely limits optical conﬁnement within the ﬁlms. MgO
(n = 1.7 at 1550 nm) provides a large refractive index
contrast, but the severe lattice mismatch with BTO leads
to poor crystallinity and degraded EO performance14–16.
(LaAlO3)0.3-(Sr2TaAlO6)0.7 (LSAT, n = 1.99 at 1550 nm)
potentially
offers
a
more
balanced
trade-off,
with
improved lattice compatibility and better index mismatch.
Large-size (up to 3 inch) LSAT wafers are also commer-
cially available at considerably lower cost compared with
those of rare-earth scandates17. However, compressive
strain in BTO ﬁlms grown on LSAT often induces out-of-
plane or mixed polarization which are partially switchable,
incompatible with in-plane electrode conﬁgurations used
in photonic devices18. While natural strain relaxation
favors in-plane polarization, it frequently degrades crys-
tallinity and increases surface roughness of BTO ﬁlms19,
leading to reduced EO efﬁciency and higher optical losses.
Our previous studies have demonstrated that buffer layers
with lattice parameters closely matched to BTO can pro-
mote in-plane domain conﬁguration and enhance EO per-
formance20.
Here,
we
propose
a
more
advanced,
heterogeneous buffer-free approach that leverages intrinsic
lattice mismatch to engineer local structural distortions and
a multiphase architecture, thereby further enhancing the
EO response of BTO ﬁlms. This strategy is inspired by lead-
free ferroelectrics such as (K,Nb)NbO3, where multiphase
coexistence near the polymorphic phase boundary (PPB)
results in polar nanoregions imbued with orthorhombic and
tetragonal
phases,
enabling
ultrahigh
piezoelectric
response21–23. Traditionally, such multiphase boundaries
are created through intricate compositional modiﬁcations.
However, comparable boundaries can also be achieved by
reducing crystal symmetry via local structural distortions,
without the need for complex compositional tuning24. A
similar hypothesis suggests that domain-wall-associated
polar nanoregions, particularly those stabilized by strain-
induced structural transitions, play a key role in enhancing
the functional properties of the material. This is supported
by theoretical simulations, which attribute the enhanced EO
response in strained BTO ﬁlms to the Ising–Néel transition
and the emergence of pseudo-orthorhombic phases at 90°
domain walls25. Our approach thus highlights the potential
of strain-engineered domain-wall structures in enabling
functional enhancement in ferroelectric thin ﬁlms.
Building on this strategy, we introduce a self-buffer
layer for epitaxial growth of BTO on LSAT, achieving a
Pockels coefﬁcient γ42 exceeding 358 pm V⁻¹ and elevating
the Curie temperature from 120 °C (bulk) to 200 °C.
Leveraging
this
high-performance
EO
platform,
we
demonstrate the ﬁrst on-chip BTO electro-optic mod-
ulator on an LSAT oxide insulator, with a half-wave
voltage length product VπL of 0.7 V cm (1550 nm) and a
6 dB EO bandwidth of ~28 GHz. Crucially, the ﬁlms are
fabricated via a ﬂexible, cost-effective single physical
vapor deposition process (as demonstrated here via
pulsed-laser deposition), eliminating the need for wafer
bonding or complex post-growth treatments. Moreover,
our strain-engineered domain structure design can be
readily transferred to other emerging ferroelectric per-
ovskites to tune the balance between lattice mismatch and
thin-ﬁlm performance, opening a new pathway to multi-
functional integrated photonics.
Results
Periodic lateral strain engineering
Epitaxial
growth
of
BTO
thin
ﬁlms
on
lattice-
mismatched substrates involves a complex interplay
between strain relaxation, domain structure evolution and
crystallinity, all of which collectively determine the
resulting
electro-optic
properties26.
On
highly
mis-
matched substrates (e.g., MgO), strain relaxation through
various
forms
of
dislocations
promotes
a
three-
dimensional island growth mode, compromising the
surface
morphology
and
crystallinity
and
thereby
degrading the electro-optic performance14–16. In contrast,
layer-by-layer or step-ﬂow growth modes preserve epi-
taxial relationships and yield high-quality ﬁlms. However,
they typically impose compressive strain constraints that
tend to stabilize c-oriented BTO domains, thereby sup-
pressing in-plane polarization responses that are essential
for large electro-optic coefﬁcients27.
To address this trade-off, we introduce a strain mod-
ulation strategy for periodic structures design, and the
schematic diagram of the resulting domain conﬁguration
is illustrated in Fig. 1a. At the early stage of ﬁlm growth on
compressive substrates, periodic nucleation sites for dis-
locations are introduced, allowing localized lattice regions
above these sites to undergo strain relaxation and thereby
stabilize into a-domains. Meanwhile, adjacent regions
remain
strained
and
consequently
stabilize
into
c-domains. Therefore, this process leads to a laterally
periodic a/c domain conﬁguration. Unlike conventional
BTO ﬁlms where strain relaxation typically completes
within a thickness of ~40 nm28, this built-in lateral strain
variation can be sustained over greater thicknesses
through a well-controlled step-ﬂow growth mode. Phase-
ﬁeld simulations (see Fig. S5 and Supplementary Infor-
mation Note 5) in Fig. 1b further demonstrate that the
Deng et al. Light: Science & Applications (2026) 15:21 
Page 2 of 10

<!-- page 3 -->
engineered a/c domain conﬁguration induces the forma-
tion of transitional regions near the domain walls, char-
acterized by polar nanoregions with an orthorhombic (O-)
phase. Notably, such structural features bear a strong
resemblance to the multiphase coexistence observed near
polymorphic phase boundaries (PPBs) in (K,Nb)NbO3,
where the interplay between tetragonal (T-) and O-phase
leads to enhanced dielectric and piezoelectric responses
due to facilitated local polarization rotation dynam-
ics21–24,29. The effectiveness of our approach is further
substantiated by comparing the measured EO coefﬁcients
of our BTO ﬁlms with reported values for ﬁlms fabricated
by different deposition methods on various substrates
(Fig. 1c, see Supplementary Information Note 1 for
references)12,15,16,20,30–38. While BTO ﬁlms on substrates
with severe lattice mismatch often suffer from signiﬁcant
performance degradation, our ﬁlms exhibit substantially
enhanced Pockels coefﬁcients. These results underscore
the effectiveness of our strategy in maintaining high
crystallinity
while
introducing
functionally
beneﬁcial
polar nanostructures, thereby overcoming the limitations
of conventional strain engineering.
As previously discussed, it is crucial to facilitate early-
stage strain relaxation and ensure high crystallinity by
600
500
400
300
200
100
0
Pockels electro-optic coefﬁcient (pm V–1)
6
4
2
0
–2
–4
Nominal mismatch (%)
f
This work
BTO Bulk
LiNbO3
LSAT
STO/SiO2/Si
BSO/STO
PLD
MBE
Sol-gel
RFMS
MOCVD
Method
MgO
Ref.20
La2O2CO3 /SiO2
r42
rc
Sénarmont
method 
Extracted 
from device
r42
rc
Intensity (Arb. unit)
45.0
44.5
Process
Primary  (48 nm)
Modiﬁed (45 nm)
Modiﬁed (180 nm)
Intensity (Arb. unit)
49
48
47
46
45
2θ (°)
2θ (°)
44
43
42
Process
Primary 
Modiﬁed 
LSAT
002
Bulk
c-domain
Bulk
a-domain 
a
d
c
c-domain
a-domain
Polymorphic phase boundary
b
PPB-like region
PPB-like region
200 nm
400
0
pm
1
0
–1
Arb. unit
e
Strained
–400
Dislocations
Fig. 1 Process-guided structural engineering of BaTiO3 (BTO) thin ﬁlms for enhanced electro-optic performance via periodic in-plane
strain. a Schematic of the structural conﬁguration in BTO thin ﬁlms fabricated using the modiﬁed process, in which alternating a- and c-domains
form polymorphic phase boundaries. b Phase-ﬁeld simulation results showing the emergence of orthorhombic phase (O-phase) polar nanoregions at
the domain boundaries. Arrows represent the three-dimensional polarization vectors. The background color indicates domain types: brown for c-
domains and blue for a-domains. c Comparison of Pockels electro-optic coefﬁcients for BTO ﬁlms grown on substrates with varying lattice mismatch
using different deposition methods. This work achieves a signiﬁcantly enhanced coefﬁcient of BTO thin ﬁlms grown on substrates with large
compressive mismatch. rc denotes the effective EO coefﬁcient, while r42 is an EO tensor component of BTO ﬁlms. d XRD θ–2θ scans near the (002)
reﬂection for BTO ﬁlms grown on LSAT via the primary and modiﬁed processes. e XRD θ–2θ scans for ﬁlms of varying thicknesses and processes,
indicating strain relaxation and the formation of a self-buffer layer at the initial stage of deposition in the modiﬁed process. f (Top) Atomic force
microscopy image and (bottom) projected in-plane piezoresponse image of the modiﬁed BTO ﬁlm. The projected piezoresponse is synthesized from
both phase and amplitude signals
Deng et al. Light: Science & Applications (2026) 15:21 
Page 3 of 10

<!-- page 4 -->
tailoring the growth conditions. As presented in Fig. 1d,
θ–2θ X-ray diffraction (XRD) patterns reveal clear dis-
tinctions between ﬁlms grown under the primary and
modiﬁed processes (see details in Method). The ﬁlm
prepared by the primary process displays a sharp (002)
reﬂection that coincides with the position of c-domain in
bulk BTO, indicating a compressively strained c-domain
structure. In contrast, the modiﬁed process results in a
high-angle shifted (002) peak, suggesting a reduced out-
of-plane lattice parameter (~4.0 Å) and a quasi-a-domain
state through the strain relaxation. To investigate the
evolution
of
strain
relaxation
during
growth,
thickness-dependent XRD were performed (Fig. 1e). For
the 48 nm-thick ﬁlm grown by the primary process, the
(002) peak posited at lower angles conﬁrms a substantial
out-of-plane lattice expansion due to the compressive
strain, in contrast with the 45 nm-thick ﬁlm grown by the
modiﬁed process. The high-angle (002) peak of the latter
indicates that the strain has largely been relaxed during
the initial growth stage. As the ﬁlm thickness increases to
180 nm, the diffraction peak remains sharp with only a
minor shoulder. This suggests that a relaxed interfacial
layer formed during the early stage acts as a self-buffer,
enabling subsequent high-quality epitaxial growth with
minimal strain accumulation, and maintains the overall
high crystallinity with a rocking curve width of 0.072°
(shown in Fig. S1). The reciprocal space maps (RSMs)
shown in Fig. S2 further support the presence of a mixed-
domain structure. The mixed a- and c-domains lead to a
diffused proﬁle of the BTO (103) reﬂection, indicating the
coexistence of multiple strain states stabilized by the
modiﬁed growth process.
Topographic
and
piezoresponse
force
microscopy
(PFM) images presented in Fig. 1f conﬁrm the excellent
crystalline quality and characteristic domain conﬁgura-
tion of the ﬁlm. The 180 nm-thick ﬁlm exhibits atomically
smooth surfaces with a root-mean-square roughness of
0.2 nm. The projected piezoresponse image reveals curvy
and diffused domain boundaries, in contrast to the shar-
ply deﬁned patterns typical of conventional ferroelectrics,
indicating the presence of complex polarization distribu-
tion underlain by the a/c domains and intermediate
O-phase nanoregions.
Figure
2
presents
scanning
transmission
electron
microscopy (STEM) results for the self-buffered BTO
ﬁlms, revealing the evolution of lattice modulations along
both the growth direction and the in-plane axis. The low-
magniﬁcation high-angle annular dark-ﬁeld (HAADF-
STEM) image (Fig. 2a) displays a well-deﬁned bilayer
structure. A ~40 nm-thick bottom region exhibits distinct
contrast features compared to the upper layer, identiﬁed
as the self-buffer layer. Fig. S3 shows a sharp atomic
interface between the ﬁlm and substrate, where the ~3%
lattice mismatch between LSAT and BTO is relieved by
periodic edge dislocations (approximately one every
15 nm), as expected for mismatch compensation. The
upper part of the ﬁlm exhibits pronounced stripe-like
lateral modulations. These periodic contrast variations,
oriented along the in-plane direction, suggest the emer-
gence of a spatially ordered lattice distortion (see Fig. S3
and Supplementary Information Note 2).
To further quantify the periodic structural modulations,
nanobeam electron diffraction (NED) was performed, as
depicted in Fig. 2b. The region above the buffer layer reveals
well-deﬁned stripe-like contrast along the in-plane direc-
tion, indicative of a periodic in-plane strain modulation that
persists throughout the remaining ﬁlm thickness. The out-
of-plane strain component (εzz) displays a markedly differ-
ent behavior. Near the substrate interface, a pronounced
lattice expansion is observed, consistent with strong out-of-
plane strain imposed by substrate clamping. However, the
εzz component becomes laterally uniform above the self-
buffer layer, showing little variation along the x direction
and lacking the periodic features observed the in-plane
strain component (εxx). Statistical analysis of the average
lattice parameters within selected boxed regions conﬁrms a
clear transition across the self-buffer boundary. The in-
plane and out-of-plane lattice constants show signiﬁcant
disparity below ~40 nm, whereas they gradually converge
above this boundary (Fig. 2c), consistent with the visual
boundary of the self-buffer region. Furthermore, line pro-
ﬁles extracted along the in-plane direction at various depths
consistently show periodic variations in the in-plane lattice
parameters (Fig. 2d).
To establish the correlation between the strain mod-
ulation and polarization distribution, local polarization
vectors were extracted from atomically resolved HAADF-
STEM images. Figure 2e presents atomic-resolution
images acquired from representative regions near the
self-buffer layer (P1) and c/a-domain boundaries (P2). In
the P1 region, strain relaxation near the self-buffer layer
induces the formation of a-domains accompanied by a
high density of O-phase regions polarized along the <101>
directions. Meanwhile, the P2 region reveals a more
complex polarization distribution, featuring both in-plane
and out-of-plane components as well as distinct O-phase
polar nanoregions. The polarization vectors in these
nanoregions
gradually
bridge
the
adjacent
a-
and
c- domains (T-phase). This intermediate-phase char-
acterizes
the
presence
of
local
polymorphic
phase
boundaries and suggests an enhanced polar instability
near the domain walls. The emergence of such complex
polar textures and domain wall geometries highlights the
strong coupling between the lateral strain modulation and
polarization
response.
Notably,
the
experimentally
observed features are in qualitative agreement with the
phase-ﬁeld simulated polarization patterns in Fig. 1b and
Fig. S5.
Deng et al. Light: Science & Applications (2026) 15:21 
Page 4 of 10

<!-- page 5 -->
Taken together, the formation of a self-buffer layer
relieves the substrate-induced strain, enabling the stabi-
lization of a quasi-periodic lateral strain ﬁeld in the upper
layers. This in-plane strain modulation, in turn, governs
the formation and distribution of ferroelectric domains
and promotes the emergence of rotationally distorted
polar structures near domain walls.
Electro-optic enhancement via polymorphic nanoregions
We employed a home-designed Sénarmont system (Fig.
3a) to characterize electro-optic performance of the BTO
ﬁlms39,40. Under an applied DC bias, the polarization
switching dynamics of domains oriented along the <110>
direction yield a pronounced EO hysteresis loop (Fig. 3b),
consistent with ferroelectric switching behavior. The
relatively low coercive ﬁeld observed is attributed to the
presence of polymorphic phase nanoregions, which lower
the energetic barrier for in-plane polarization rotation.
The EO response measured under AC excitation displays
a strong crystallographic anisotropy. As shown in Fig. 3c,
a peak effective EO coefﬁcient of rc = 253 pm V⁻¹ is
obtained when the electric ﬁeld is applied along the <110>
direction,
while
a
signiﬁcantly
lower
value
of
rc = 34 pm V⁻¹ is observed under an electric ﬁeld along
the <100 > . This anisotropy arises from the inherent form
of the EO tensor of tetragonal BTO. Speciﬁcally, the large
EO response along the <110> can be ascribed to the
projection of the intrinsic r42 coefﬁcient, which is calcu-
lated to be 358 pm V⁻¹ according to Supplementary
Information Note 4. Compared with previous studies20,
this enhancement is attributed to the rational structural
design that facilitates the formation of O-phase nanodo-
mains and local PPBs.
To identify the role of intermediate O-phase nanor-
egions, we performed in-situ temperature-dependent
structural
characterizations
(see
Supplementary
a-domain
O-Phase
–4
–2
0
2
Strain (%)
140
120
100
80
60
40
20
0
Z axis (nm)
In-plane
Out-of-plane
–1.0
0.0
1.0
Strain (%)
In-plane
Out-of-plane
–1.0
0.0
1.0
Strain (%)
120
100
80
60
40
20
0
X axis (nm)
In-plane
Out-of-plane
c-domain
a2-domain
a1-domain
O-Phase
a
Self-buffer layer
50 nm
P1
P2
In-plane
Out-of-plane
z
x
c
b
d
2 nm
P1
P2
25 nm
z
x
e
LSAT
BTO
4.100
4.075
4.050
4.025
4.000
3.975
3.950
3.925
Å
Fig. 2 Microscopic analysis of the strain and polarization states in self-buffered BTO thin ﬁlms. a Cross-sectional low-magniﬁcation STEM
image of the BTO ﬁlm. A ~40 nm-thick self-buffer layer is observed at the bottom, distinct from the upper region exhibiting lateral contrast
modulations. b Strain maps for the in-plane (up) and out-of-plane (down) components, reconstructed from the NED results. c Strain proﬁles of the
average in-plane and out-of-plane strain components along the growth z (<001>) direction. d Lateral strain proﬁles extracted at different depths,
conﬁrming the emergence of periodic in-plane strain modulations above the buffer layer. The line color corresponds to the regions selected in b.
e Atomic-resolution HAADF-STEM images for two representative regions, showing (up) O-phase regions near the self-buffer layer and (down)
complex polar structures at a domain boundary. The reconstructed polarization vectors are overlaid
Deng et al. Light: Science & Applications (2026) 15:21 
Page 5 of 10

<!-- page 6 -->
Information Note 6). As shown in Fig. 3d, second har-
monic generation (SHG) intensity decreases almost line-
arly
from
−150 °C
to
100 °C,
without
any
abrupt
discontinuity. Since SHG is highly sensitive to macro-
scopic lattice symmetry breaking, the absence of abrupt
changes suggests that no long-range phase transition
occurs within this temperature range41. Interestingly, a
noticeable dip in the SHG intensity is observed near
120 °C, indicative of local polarization vector rearrange-
ments. Further insights are provided by the temperature-
dependent XRD. As shown in Fig. 3d, the out-of-plane
lattice constant increases linearly with temperature above
3
2
1
0
SHG intensity (×104 arb. unit)
300
200
100
0
–100
Temperature (°C)
Temperature (°C)
4.032
4.028
4.024
4.020
Lattice constant (Å)
100
80
60
40
20
0
O fraction (%)
–150
–100
–50
0
50
100
150
2.0
1.0
0.0
1.2
1.0
0.8
0.6
0.4
0.2
E along <110>
E along <100>
–1.0
–0.5
0.0
0.5
1.0
Δn (×10–3)
Δn (×10–3)
–5
–2
0
2
EDC (V μm–1)
EAC (V μm–1)
5
a
b
c
Quarter-Wave 
Plates
Polarizer
Half-Wave 
Plates
rc=253 pm V–1
rc=34 pm V–1
XRD
SHG
Cubic
Tc
e
–142 °C
O-phase
7 °C
157 °C
d
8
6
4
2
0
Distribution amplitude (Arb. unit)
6
5
4
3
2
1
0
0.8
0.6
–6.00
–6.05
–6.10
–6.15
–6.20
–6.25
0.04
E–1 (cm kv–1)
E–1 (cm kv–1)
0.08
0.4
0.2
0.0
Distribution amplitude (Arb. unit)
–8
–7
–6
–5
–4
1.0
0.5
0.0
State
–6
–4
–2
0
2
log(ton) (s)
log(ton) (s)
log(ton) (s)
Pulse voltage
50 V
40 V
30 V
20 V
10 V
g
α1= 3.09 kV cm–1
α2= 3.42 kV cm–1
f
0.08
0.04
3.75
3.70
3.65
3.60
3.55
3.50
90
180
270
.0
180
270
.0
90
90
180
270
.0
log(tmean)
log(tmean)
O
O
O
c
a
Fig. 3 Enhanced electro-optic response in self-buffered BTO ﬁlms via polymorphic phase nanoregions. a Schematic illustration of the optical
setup for EO measurements. b Changes in the refractive index as a function of DC bias. AC ﬁeld = 0.625 V μm−1. c Changes in the refractive index as
a function of AC electric ﬁeld applied along the <110> and <100> in-plane directions. DC ﬁeld = 3.125 V μm−1. The effective EO coefﬁcients are
extracted from the slopes. d Temperature-dependent evolution of second-harmonic generation (SHG) intensity and out-of-plane lattice constant
obtained from XRD. Inset: schematic illustration of domain evolution with temperature, where “a, c, o” denotes T-phase a-/c-domain and O-phase
domain, respectively. e Temperature dependence of the O-phase fraction, extracted by ﬁtting the SHG polar patterns. Inset: SHG polarimetry ﬁtting
results at representative temperatures, with green-shaded regions indicating the contribution of the O-phase. f Normalized EO response measured
after application of various numbers of electric pulses superimposed on a 10 V DC bias, revealing ferroelectric domain switching kinetics. The states
“0” and “1” correspond to the two extreme EO responses measured in the BTO ﬁlms poled using negative and positive electric ﬁelds, respectively.
g Domain switching time extracted using a constrained nucleation model, ﬁtted with two-segment Lorentzian distributions. Inset: linear ﬁt used to
extract the activation ﬁeld α
Deng et al. Light: Science & Applications (2026) 15:21 
Page 6 of 10

<!-- page 7 -->
200 °C, characteristic of a paraelectric phase. This tem-
perature region can thus be assigned as the Curie tem-
perature Tc. Below 200 °C, the lattice constant clearly
deviates from linear thermal expansion, reﬂecting the
presence of distinct domain responses. From 0 °C to
120 °C, the out-of-plane lattice constant decreases with
increasing temperature, consistent with a bulk-like ther-
mal contraction behavior of the mixed a- and c-domains
toward the cubic phase. The thermal expansion recovers a
positive trend between 120 °C and 200 °C. Although the
elevated Tc has traditionally been ascribed to epitaxially
constrained c-domains42, it is also necessary to consider
the potential inﬂuence of polymorphic phase nanoregions.
Unlike classical long-range ordered ferroelectric systems,
these nanoregions may give rise to spatially localized polar
instabilities, thus resulting in a broadened structural
transition behavior.
To further elucidate the structural evolution behaviors,
we performed temperature-dependent SHG polarimetry
(Fig. 3e). By virtue of the symmetry sensitivity of SHG
patterns, the O-phase fraction can be extracted by ﬁtting
the angular response43. As shown in Fig. 3e, the O-phase
ratio remains nearly constant from room temperature to
160 °C. This temperature-invariant behavior indicates that
the intermediate O-phase nanoregions are structurally
robust and thermally stable. These ﬁndings corroborate
the hypothesis that both the diffuse phase transition and
the elevated Curie temperature arise from the persistent
polymorphic nanoregions, rather than from an abrupt
symmetry breaking. Altogether, the results reveal non-
classical ferroelectric behavior in self-buffered BTO ﬁlms
due to the nanoscale structural heterogeneity.
The domain switching kinetics were examined via
pulsed EO and SHG mapping measurements under <110>
electric ﬁelds (see Fig. S4 and Supplementary Information
Note 3). The coexistence of a-/c-domains and O-phase
nanoregions provides a continuous pathway for polar-
ization rotation, leading to switching behaviors that
deviate from classical phenomenological models such as
the nucleation-limited switching (NLS) model44,45. As
shown in Fig. 3f, the switching kinetics under different
pulse voltages can be well described by a superposition of
two distinct NLS components. The fast component is
attributed to the in-plane switching of a-domains, while
the slower one likely arises from the in-plane reorienta-
tion of c-domains. Previous studies showed that the
substrate elastic clamping effect in epitaxial ﬁlms can
suppress the polarization reorientation of c-domains46.
However, as shown in Fig. 3g, the Lorentzian distributions
of switching time here extracted from the two-stage NLS
model reveal that the activation ﬁelds (α) associated with
both the fast and slow components are comparable. This
thus indicates a reduction in the switching energy barrier
for the c-domains, due to the presence of intermediate
O-phase nanoregions which facilitate polarization rota-
tion pathways via local O-T structural transitions. Overall,
these results highlight the distinctive complex switching
kinetics of self-buffered BTO ﬁlms dictated by the engi-
neered PPBs.
On-chip electro-optic modulator
To evaluate the viability of our BTO on LSAT as a new
photonic platform, we fabricated a Mach–Zehnder inter-
ferometer (MZI) modulator featuring two 50:50 Y-branch
splitters and a pair of 1-mm-long phase-shifting arms (Fig.
4a). One arm incorporates a coplanar waveguide traveling-
wave electrode in a ground–signal–ground (GSG) layout,
delivering both a radio-frequency signal and a tunable DC
bias (see Supplementary Information Note 7). This bias
stabilizes the ferroelectric polarization of the BTO layer,
ensuring a robust electro-optic response. The other arm
employs a ground–signal (GS) capacitive electrode driven
by a low DC bias, which induces a ﬁne phase shift to tune
the optical operating point.
Figure 4b shows the cross-sectional view of the mod-
ulator, where Si3N4 strip-loaded waveguides conﬁne light
within the BTO layer. Finite-element simulations (Fig. 4c)
conﬁrm that the waveguide supports a single transverse
electric (TE) mode, with the optical ﬁeld predominantly
conﬁned to the 300-nm-thick BTO ﬁlm (see Supple-
mentary Information Note 8). The electrodes spaced
5.5 μm apart generate a nearly uniform in-plane electric
ﬁeld across the active layer, achieving a strong electric-
optic ﬁeld overlap. Based on the electro-optic tensor of
BTO (Supplementary Information Note 4), only the TE
mode is effectively modulated under an applied ﬁeld along
the <110> crystallographic direction. The electric-optic
ﬁeld overlap factor is calculated to be 45%, corresponding
to a theoretical half-wave voltage–length product (VπL) of
0.73 V cm. The low-frequency modulation response at
1550 nm (Fig. 4d) yields a half-wave voltage of 7 V. This
corresponds to a VπL of 0.7 V cm, in close agreement with
the simulated value. In comparison with previously
reported BTO-based MZI modulators fabricated on other
substrates, where the VπL values were either directly
measured47–49 or calculated from wavelength shift8,32,50,
our device demonstrates solidly competitive performance.
Figure 4e further illustrates the electro-optic bandwidth
of the device. The frequency response of EO S21 remains
relatively ﬂat, with a 3 dB bandwidth of approximately
12 GHz. A clear roll-off is observed around 12 GHz, pri-
marily due to the lack of group velocity matching between
the optical and electrical signals. The group velocity
mismatch can be mitigated through further optimization
of the electrode and the optical waveguide structures.
Nevertheless, the device exhibits a substantially larger EO
bandwidth compared to previously reported BTO MZI
demonstrations8,49,50.
Additionally,
a
relatively
ﬂat
Deng et al. Light: Science & Applications (2026) 15:21 
Page 7 of 10

<!-- page 8 -->
response is observed between 15 GHz and 28 GHz, with a
6 dB EO bandwidth located at 28 GHz, approaching the
best current devices with a reported 6 dB bandwidth of 40
GHz48. The reﬂection coefﬁcient S11 ﬂuctuates around
−15 dB, indicating a good impedance matching. In con-
clusion, we present the ﬁrst integration of a BTO-based
EO modulator on an LSAT insulating substrate. This
device demonstrates both low driving voltage and broad
operational bandwidth, underscoring the potential of the
oxide-integrated BTO platform for high-speed photonic
system applications.
Discussion
In summary, we have demonstrated a heterogenous
buffer-free, strain-engineered strategy for the epitaxial
growth of BTO ﬁlms on oxide insulator substrates,
effectively addressing the trade-off between optical con-
ﬁnement
and
lattice
compatibility
for
photonic
integration of ferroelectric thin ﬁlms. The formation of a
self-buffer layer induces lateral strain modulation, stabi-
lizing periodic a/c domain conﬁgurations and orthor-
hombic phase nanoregions. This structural modulation
signiﬁcantly enhances both the electro-optic response and
thermal
stability,
yielding
a
Pockels
coefﬁcient
r42
exceeding 358 pm V⁻¹ and a Curie temperature of 200 °C.
Based on this material platform, we present the ﬁrst
integrated MZI modulator employing epitaxial BTO on
LSAT, achieving a low VπL of 0.7 V cm and an electro-
optic bandwidth of 28 GHz. These results provide new
insights into electro-optic ﬁlm design and establish a
practical route toward high-performance modulators
using ﬂexible and cost-effective fabrication processes in
integrated photonics.
Materials and methods
Sample preparation
Epitaxial BTO ﬁlms were deposited on (001) LSAT
substrates via pulsed laser deposition (KrF excimer laser,
248 nm). For the primary deposition condition, the ﬁlm
was deposited at 680 °C in an oxygen pressure of 5 Pa, at a
laser
repetition
rate
of
3 Hz
and
laser
ﬂuence
of
1.2 J cm−2. Under modiﬁed conditions, the substrate
temperature and oxygen partial pressure were varied
within the ranges of 650–700 °C and 5–10 Pa, respectively,
while monitoring reﬂection high-energy electron diffrac-
tion (RHEED, STAIB Instruments) to ensure clear dif-
fraction
spots
and
layer-by-layer
oscillations
under
constant laser energy and frequency. Following the initial
40 nm of growth, the temperature and oxygen pressure
were stabilized at 660 °C and 10 Pa, respectively, for the
–20
–15
–10
–5
0
5
Frequency response (dB)
30
25
20
15
10
5
0
Frequency (GHz)
EO S21
EE S11
6
4
2
0
Voltage (V)
<110>
G
G
S
200 μm
<110>
G
G
S
G
S
a
b
c
d
e
12 GHz
28 GHz
1 μm
Si3N4
BTO
LSAT
2 μm
Au
Air
BTO
LSAT
Si3N4
Vπ = 7 V
Vπ L = 0.7 V cm
1
0.5
0
Normalized transmission
(Arb. unit)
 ¯
Fig. 4 BTO on LSAT as an oxide platform for integrated high-performance electro-optic modulation. a Optical micrograph of the fabricated
Mach–Zehnder interferometer (MZI) based modulator, implemented on self-buffered BTO ﬁlm in conjunction with Si3N4 strip waveguides. Gold
electrodes conﬁgured in a ground–signal–ground (GSG) layout are used for high-speed driving. b Tilted-view scanning electron microscopy (SEM)
image of the edge-coupling interface. The brown region corresponds to the Si3N4 waveguide, the blue region to the BTO ﬁlm, and the green region
to the LSAT substrate. c Simulated mode proﬁles of the photonic TE mode (top) and quasi-static electric ﬁeld (bottom) within the modulator cross-
section. d Normalized optical transmission as a function of the applied voltage. The extracted half-wave voltage (Vπ) is 7 V, corresponding to a half-
wave voltage-length product (VπL) of 0.7 V cm. e Frequency response of the MZI modulator, showing the electro-optic bandwidth and the
impedance matching condition of the microwave input
Deng et al. Light: Science & Applications (2026) 15:21 
Page 8 of 10

<!-- page 9 -->
remainder of the deposition. After deposition, all ﬁlms
were annealed in situ for 10 min under an oxygen pres-
sure of 20 kPa.
STEM
The STEM images and the 4D datasets were acquired
on a probe-aberration-corrected FEI Titan Cubed Themis
G2 microscope operated at 300 kV. 4D datasets were
acquired with a pixel array detector EMPAD. The con-
vergence semi-angle was set to 0.83 mrad. Each diffrac-
tion pattern has a dimension of 128 × 128 pixels, and the
camera length is 360 mm, giving a reciprocal pixel size of
0.043 Å−1.
Fabrication of BTO-on-LSAT photonic integrated circuit
A 260 nm-thick Si3N4 layer was ﬁrst deposited over the
BTO ﬁlm on LSAT substrate by plasma-enhanced che-
mical vapor deposition (PECVD) to serve as the core
material for the subsequent strip waveguide. The photo-
nic waveguide pattern was deﬁned by an electron-beam
lithography system (EBPG 5200, Raith). Fluorine-based
reactive ion etching (RIE) was employed to form the Si3N4
waveguides using the electron-beam resist as the etch
mask. After waveguide fabrication, a >2 μm-thick pat-
terned optical isolation layer was formed using benzocy-
clobutene (BCB) photoresist. The electrode patterns were
subsequently deﬁned by a UV laser direct-write litho-
graphy system (MicroWriter ML3, Durham Magneto
Optics), followed by e-beam evaporation of a 10 nm Ti
adhesion layer and a 500 nm Au layer. Lift-off was per-
formed to obtain the ﬁnal metal electrodes. The sample
was then diced and facet-polished to form low-loss edge
couplers.
Electro-optic modulator characterization
A tunable laser (TSL-570, Santec) operating in the
C-band was used for linear electro-optic tuning mea-
surements, with a ﬁber polarization controller employed to
ensure excitation of the TE mode. For Vπ measurements,
the MZI modulator was driven by a 100 kHz triangular
voltage signal while monitoring the optical transmission in
real time. Electro-optic bandwidth measurements were
conducted using a 67 GHz vector network analyzer (PNA
N5247B, Keysight) in conjunction with a Bias-T, which
enabled application of a DC bias but limited the measur-
able frequency range up to 40 GHz. A pair of high-speed
microwave probes delivered the RF signal to the input of
the transmission line, and the output was terminated with
a 50 Ω load. Optical coupling into and out of the chip was
achieved via lensed ﬁbers. The modulated optical signal
was ampliﬁed by an erbium-doped ﬁber ampliﬁer and
subsequently
detected
by
a
high-speed
photodiode,
enabling extraction of the S21 response.
Acknowledgements
This work was supported by Ministry of Education of China Scientiﬁc Research
Innovation Capability Support Project for Young Faculty under Grant
No.ZYGXQNJSKYCXNLZCXM-M17, the Basic Science Center Project of National
Natural Science Foundation of China (NSFC) under Grant No. 52388201, NSFC
grants No. U24A2009 and 12474087, Beijing Municipal Natural Science
Foundation under Grant No. JQ24011 and Z240008, and by China Postdoctoral
Science Foundation under Grant No. 2023M741873. This work is also
technically supported by Synergetic Extreme Condition User Facility (SECUF,
https://cstr.cn/31123.02.SECUF).
Author details
1State Key Laboratory of New Ceramic Materials, School of Materials Science
and Engineering, Tsinghua University, Beijing, China. 2Beijing National Research
Centre for Information Science and Technology (BNRist), State Key Laboratory
of Space Network and Communications, Department of Electronic
Engineering, Tsinghua University, Beijing, China. 3State Key Laboratory of
Silicon and Advanced Semiconductor Materials, School of Materials Science
and Engineering, Zhejiang University, Hangzhou, Zhejiang, China. 4Zhejiang
Key Laboratory of Advanced Solid State Energy Storage Technology and
Applications, Taizhou Institute of Zhejiang University, Taizhou, Zhejiang, China.
5Institute of Fundamental and Transdisciplinary Research, Zhejiang University,
Hangzhou, China. 6Beijing National Laboratory for Condensed Matter Physics,
Institute of Physics, Chinese Academy of Sciences, Beijing, China
Author contributions
Q.L. and C.-Z.S. conceptualized and oversaw this work. C.-G.D., Y.-T.H., and W.-
F.Y. designed the speciﬁc operational details of this work. C.-G.D. prepared the
samples. C.-G.D., Y.-T.H., H.L. and Z.-S.Z. prepared the photonic integrated
devices. Z.-J.H. and Y.-J.W. performed phase-ﬁeld simulations. C.-G.D., Y.-T.H.,
H.Y., Y.-P.M., H.L., H.-J.H., and W.L. carried out the measurements. Q.L., C.-Z.S.,
C.-G.D., Y.-T. H, W.-F.Y., W.L., J.M., B.X., R.Y., J.-F.L., J.Z., and Y.L. provided data
analysis. C.-G.D. and Y.-T.H. wrote the manuscript. All authors participated in
data discussions and manuscript editing.
Data availability
All data needed to evaluate the conclusions in the paper are present in the
paper and/or the Supplementary Information. Additional data can be provided
by the authors upon reasonable request.
Conﬂict of interest
The authors declare no competing interests.
Supplementary information The online version contains supplementary
material available at https://doi.org/10.1038/s41377-025-02081-9.
Received: 3 June 2025 Revised: 22 September 2025 Accepted: 2 October
2025
References
1.
Shastri, B. J. et al. Photonics for artiﬁcial intelligence and neuromorphic
computing. Nat. Photonics 15, 102–114 (2021).
2.
Marpaung, D., Yao, J. P. & Capmany, J. Integrated microwave photonics. Nat.
Photonics 13, 80–90 (2019).
3.
Hu, Y. W. et al. Integrated electro-optics on thin-ﬁlm lithium niobate. Nat. Rev.
Phys. 7, 237–254 (2025).
4.
Atabaki, A. H. et al. Integrating photonics with silicon nanoelectronics
for the next generation of systems on a chip. Nature 556, 349–354
(2018).
5.
Wang, C. et al. Integrated lithium niobate electro-optic modulators operating
at CMOS-compatible voltages. Nature 562, 101–104 (2018).
6.
Yu, M. J. et al. Integrated femtosecond pulse generator on thin-ﬁlm lithium
niobate. Nature 612, 252–258 (2022).
7.
Zhu, D. et al. Integrated photonics on thin-ﬁlm lithium niobate. Adv. Opt.
Photonics 13, 242–352 (2021).
Deng et al. Light: Science & Applications (2026) 15:21 
Page 9 of 10

<!-- page 10 -->
8.
Xiong, C. et al. Active silicon integrated nanophotonics: ferroelectric BaTiO3
devices. Nano Lett. 14, 1419–1425 (2014).
9.
Wang, H. et al. Advancing inorganic electro-optical materials for 5 G com-
munications: from fundamental mechanisms to future perspectives. Light Sci.
Appl. 14, 190 (2025).
10.
Eltes, F. et al. An integrated optical modulator operating at cryogenic tem-
peratures. Nat. Mater. 19, 1164–1168 (2020).
11.
Geler-Kremer, J. et al. A ferroelectric multilevel non-volatile photonic phase
shifter. Nat. Photonics 16, 491–497 (2022).
12.
Abel, S. et al. A strong electro-optically active lead-free ferroelectric integrated
on silicon. Nat. Commun. 4, 1671 (2013).
13.
Abel, S. et al. Large Pockels effect in micro- and nanostructured barium tita-
nate integrated on silicon. Nat. Mater. 18, 42–47 (2019).
14.
Petraru, A. et al. Ferroelectric BaTiO3 thin-ﬁlm optical waveguide modulators.
Appl. Phys. Lett. 81, 1375–1377 (2002).
15.
Wessels, B. W. Ferroelectric epitaxial thin ﬁlms for integrated optics. Annu. Rev.
Mater. Res. 37, 659–679 (2007).
16.
Kim, I. D. et al. Ridge waveguide using highly oriented BaTiO3 thin ﬁlms for
electro-optic application. J. Asian Ceram. Societ. 2, 231–234 (2014).
17.
Cao, Y. et al. A barium titanate-on-oxide insulator optoelectronics platform.
Adv. Mater. 33, 2101128 (2021).
18.
Lee, J. W. et al. In-plane quasi-single-domain BaTiO3 via interfacial symmetry
engineering. Nat. Commun. 12, 6784 (2021).
19.
Wang, T. Q. et al. Critical thickness and strain relaxation in molecular beam
epitaxy-grown SrTiO3 ﬁlms. Appl. Phys. Lett. 103, 212904 (2013).
20.
Yu, H. et al. Tuning the electro-optic properties of BaTiO3 epitaxial thin ﬁlms via
buffer layer-controlled polarization rotation paths. Adv. Funct. Mater. 34,
2315579 (2024).
21.
Huangfu, G. et al. Giant electric ﬁeld–induced strain in lead-free piezoceramics.
Science 378, 1125–1130 (2022).
22.
Liu, Q. et al. Practical high-performance lead-free piezoelectrics: structural
ﬂexibility beyond utilizing multiphase coexistence. Natl Sci. Rev. 7, 355–365
(2020).
23.
Lv, X. et al. Emerging new phase boundary in potassium sodium-niobate
based ceramics. Chem. Soc. Rev. 49, 671–707 (2020).
24.
Liu, H. J. et al. Giant piezoelectricity in oxide thin ﬁlms with nanopillar structure.
Science 369, 292–297 (2020).
25.
Li, W. T., Landis, C. M. & Demkov, A. A. Domain morphology and electro-optic
effect in Si-integrated epitaxial BaTiO3 ﬁlms. Phys. Rev. Mater. 6, 095203 (2022).
26.
Jiang, Y. et al. Enabling ultra-low-voltage switching in BaTiO3. Nat. Mater. 21,
779–785 (2022).
27.
Reitze, D. H. et al. Electro-optic properties of single crystalline ferroelectric thin
ﬁlms. Appl. Phys. Lett. 63, 596–598 (1993).
28.
Dubourdieu, C. et al. Switching of ferroelectric polarization in epitaxial BaTiO3
ﬁlms on silicon without a conducting bottom electrode. Nat. Nanotechnol. 8,
748–754 (2013).
29.
Liu, Q. et al. High-performance lead-free piezoelectrics with local structural
heterogeneity. Energy Environ. Sci. 11, 3531–3539 (2018).
30.
Zgonik, M. et al. Dielectric, elastic, piezoelectric, electro-optic, and elasto-optic
tensors of BaTiO3 crystals. Phys. Rev. B 50, 5941–5949 (1994).
31.
Bernasconi, P., Zgonik, M. & Günter, P. Temperature dependence and dis-
persion of electro-optic and elasto-optic effect in perovskite crystals. J. Appl.
Phys. 78, 2651–2658 (1995).
32.
Posadas, A. B. et al. Thick BaTiO3 epitaxial ﬁlms integrated on Si by RF sput-
tering for electro-optic modulators in Si photonics. ACS Appl. Mater. Interfaces
13, 51230–51244 (2021).
33.
Chelladurai, D. et al. Barium titanate and lithium niobate permittivity and
Pockels coefﬁcients from megahertz to sub-terahertz frequencies. Nat. Mater.
24, 868–875 (2025).
34.
Edmondson, B. I. et al. Epitaxial, electro-optically active barium titanate thin
ﬁlms on silicon by chemical solution deposition. J. Am. Ceram. Soc. 103,
1209–1218 (2020).
35.
Reynaud, M. et al. Electro-optic response in epitaxially stabilized orthorhombic
mm2 BaTiO3. Phys. Rev. Mater. 5, 035201 (2021).
36.
Petraru, A. et al. Integrated optical Mach Zehnder modulator based on
polycrystalline BaTiO3. Opt. Lett. 28, 2527–2529 (2003).
37.
Kormondy, K. J. et al. Microstructure and ferroelectricity of BaTiO3 thin
ﬁlms on Si for integrated photonics. Nanotechnology 28, 075706
(2017).
38.
Picavet, E. et al. Integration of solution-processed BaTiO3 thin ﬁlms with high
Pockels coefﬁcient on photonic platforms. Adv. Funct. Mater. 34, 2403024
(2024).
39.
Deng, C. G. et al. Reporting excellent transverse piezoelectric and electro-optic
effects in transparent rhombohedral PMN-PT single crystal by engineered
domains. Adv. Mater. 33, 2103013 (2021).
40.
Liu, X. et al. Ferroelectric crystals with giant electro-optic property enabling
ultracompact Q-switches. Science 376, 371–377 (2022).
41.
Gradauskaite, E. et al. Defeating depolarizing ﬁelds with artiﬁcial ﬂux closure in
ultrathin ferroelectrics. Nat. Mater. 22, 1492–1498 (2023).
42.
Choi, K. J. et al. Enhancement of ferroelectricity in strained BaTiO3 thin ﬁlms.
Science 306, 1005–1009 (2004).
43.
Li, W. et al. Delineating complex ferroelectric domain structures via second
harmonic generation spectral imaging. J. Mater. 9, 395–402 (2023).
44.
Nelson, C. T. et al. Domain dynamics during ferroelectric switching. Science
334, 968–971 (2011).
45.
Chen, Z. B. et al. Facilitation of ferroelectric switching via mechanical manip-
ulation of hierarchical nanoscale domain structures. Phys. Rev. Lett. 118, 017601
(2017).
46.
Xu, R. J. et al. Ferroelectric polarization reversal via successive ferroelastic
transitions. Nat. Mater. 14, 79–86 (2015).
47.
Dong, Z. M. et al. Monolithic barium titanate modulators on silicon-on-
insulator substrates. ACS Photonics 10, 4367–4376 (2023).
48.
Li, W. J. et al. Thin-ﬁlm BTO-based MZMs for next-generation IMDD
transceivers beyond 200 Gbps/λ. J. Lightw. Technol. 42, 1143–1150
(2024).
49.
Team, P. si Q. uantum A manufacturable platform for photonic quantum
computing. Nature 641, 876–883 (2025).
50.
Eltes, F. et al. A BaTiO3-based electro-optic Pockels modulator monolithically
integrated on an advanced silicon photonics platform. J. Lightw. Technol. 37,
1456–1462 (2019).
Deng et al. Light: Science & Applications (2026) 15:21 
Page 10 of 10

