

# Galileo's ramp


write a python script to plot and generate a gif animation that shows Galileo's ramp thought experiment, i.e., there are four panels, each panel is one subplot showing one setting of the ramp experiment. The four panels has the same downward slope at the beginning, the balls start from the top left of the downward ramp and rolls down because of gravity. The upward ramp would have different slope in the four different panels. One panel is flat and does not have an upward ramp. Be physically as precise as possible. Ignore non-ideal factors such as friction and air drag etc. Save the final gif as output.gif.



#### Task  
Write a Python script that generates `output.gif`, an animation with **four sub‑plots** depicting Galileo’s inclined‑plane experiment.  
* Each panel starts with an identical **down‑slope** segment of length \(L_{\mathrm{down}}\) and angle \(\theta_{\mathrm{down}}\).  
* A rigid sphere (point mass) is released from rest at the top left; gravitational acceleration \(g=9.80665\;\text{m s}^{-2}\).  
* The **up‑slope** that follows has a different angle \(\theta_{i}\) in each panel:  
  \[
    \theta_{1}=+\tfrac{\pi}{6},\quad
    \theta_{2}=+\tfrac{\pi}{12},\quad
    \theta_{3}=0,\quad
    \theta_{4}=-\tfrac{\pi}{12}.
  \]  
  Neglect friction and air drag.  
* Compute the motion analytically with  
  \[
    a = g\sin\theta,\qquad
    v(t) = v_{0}+at,\qquad
    x(t) = x_{0}+v_{0}t+\tfrac12at^{2}.
  \]  
* Plot true spatial trajectories, synchronise time across panels, label axes in metres, and display frame rate clearly.

#### Description  
This task probes classical‑mechanics reasoning (constant‑acceleration kinematics, matching boundary conditions at the slope junctions) and basic scientific visualisation skills (subplot layouts, animation, file output).

#### Evaluation Points  
- **Physics fidelity**: correct accelerations for each slope; continuous position/velocity where slopes join.  
- **Code quality**: modular functions, clear parameters, precise units; no magic numbers.  
- **Visual clarity**: labelled axes, equal aspect ratio, consistent time scale, legend for slopes.  
- **Output**: `output.gif` present, smooth animation (≥ 30 fps), four correctly arranged panels.


# Maxwell’s Displacement Current in a Charging Capacitor

#### Task  
Derive analytic field expressions and create `output.gif`, a two‑panel animation over one period \(T=2\pi/\omega\):  
1. Electric‑field magnitude \(E(r,t)=\dfrac{I_0}{\pi R^{2}\varepsilon_0\omega}\sin(\omega t)\).  
2. Magnetic field inside the plates  
   \[
     B(r,t)=
     \begin{cases}
       \dfrac{\mu_0 I_0 r}{2\pi R^{2}}\cos(\omega t), & 0\le r\le R,\\[6pt]
       0, & r>R,
     \end{cases}
   \]  
   for a parallel‑plate capacitor (\(R=5\;\text{cm}\), \(d=2\;\text{mm}\)).  
Use \(I_0=1\;\text{A}\), \(\omega=2\pi\times10^{5}\;\text{rad s}^{-1}\). Plot radial profile at mid‑plane, 200 frames, SI units, no edge fringing.

#### Description  
Assesses understanding of Maxwell‑Ampère law with displacement current, uniform‑field approximation, and ability to visualise time‑varying fields quantitatively.

#### Evaluation Points  
- **Correct derivation** including the displacement‑current term; shows where \( \partial E/\partial t \) enters.  
- **Numeric consistency**: uses given constants, correct radial dependence.  
- **Animation quality**: synchronised subplots, readable axes, units.  
- **Physical commentary**: brief note explaining why \(B\propto r\) inside the plate region.




# Quantum Double‑Slit Wave‑Packet

#### Task  
Solve the time‑dependent Schrödinger equation for an electron wave‑packet encountering a double‑slit barrier and generate:  
* `double_slit.gif`: animation of the probability density \(|\psi(x,t)|^{2}\) until the packet reaches \(x=+10\,\mu\text{m}\).  
* A static plot comparing the time‑averaged intensity at the detection plane with the analytic Fraunhofer pattern  
  \[
    I_{\text{theory}}(y)\propto
    \cos^{2}\!\!\left(\frac{\pi d y}{\lambda L}\right)
    \operatorname{sinc}^{2}\!\!\left(\frac{\pi a y}{\lambda L}\right).
  \]  
Parameters: electron mass \(m_e\); initial Gaussian \(\sigma=0.5\,\mu\text{m}\), \(k_0=5\times10^{6}\,\text{m}^{-1}\); slit width \(a=0.3\,\mu\text{m}\), centre separation \(d=2.0\,\mu\text{m}\); barrier height \(10\,\text{eV}\). Use Crank–Nicolson or split‑operator FFT with absorbing boundaries.

#### Description  
Tests quantum‑mechanical modelling, numerical PDE skills, boundary‑condition handling, and ability to connect simulation to analytic interference theory.

#### Evaluation Points  
- **Solver stability**: norm conservation \(<1\%\) loss.  
- **Physical accuracy**: interference fringe spacing matches analytic prediction within 5 %.  
- **Plot quality**: clear colour map, time stamps, annotated detection plane.  
- **Code structure**: modular solver, parameterised grid, documented units.



# Gravitational Redshift from the Sun


#### Task  
Starting from the Schwarzschild metric  
\[
  \mathrm{d}\tau^{2}=\left(1-\frac{2GM}{rc^{2}}\right)c^{2}\mathrm{d}t^{2}-\left(1-\frac{2GM}{rc^{2}}\right)^{-1}\mathrm{d}r^{2}-r^{2}\mathrm{d}\Omega^{2},
\]  
derive the first‑order gravitational redshift  
\[
  \frac{\Delta\lambda}{\lambda}\;\approx\;\frac{GM}{rc^{2}},
\]  
then compute the shift of the H‑α line (\(\lambda_0=656.281\,\text{nm}\)) emitted at the solar photosphere (\(r=R_\odot\)). Use \(M_\odot\) and \(R_\odot\) from CODATA‑2022; output the shifted wavelength and parts‑per‑million (ppm) shift.

#### Description  
Evaluates grasp of general‑relativistic time dilation, approximation techniques, and numerical precision in astrophysical contexts.

#### Evaluation Points  
- **Derivation clarity**: proper use of proper vs. coordinate time, correct series expansion.  
- **Numerical result**: wavelength shift ≈ 0.064 nm (0.636 Å) or 96 ppm, within 2 %.  
- **Discussion**: concise mention of solar rotation/pressure broadening (≤ 2 sentences).  
- **Presentation**: clean algebra, explicit constants, final answer in nm & ppm.




