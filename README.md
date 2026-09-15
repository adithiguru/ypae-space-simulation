# TOI-1338 Circumbinary System: Desert Planet & Double Sunset Simulation

A scientifically grounded, cinematic simulation of the real observed binary star system **TOI-1338** and a hypothetical Earth-sized desert planet orbiting its circumbinary barycenter, built in Python + Taichi.

## Scientific Implementation

### OBSERVED
* **TOI-1338 Stellar Parameters and Binary Orbit**:
  * **TOI-1338 A**: $1.127\,M_\odot$, F8V primary, $T_{\text{eff}} \approx 6050\,\text{K}$.
  * **TOI-1338 B**: $0.299\,M_\odot$, M-dwarf secondary, $T_{\text{eff}} \approx 3200\,\text{K}$.
  * **Orbit**: $a_{\text{bin}} = 0.1321\,\text{AU}$, $e_{\text{bin}} = 0.1555$, $P \approx 14.61\,\text{days}$.
  *(Baseline parameters match the ESPRESSO/SB2 dynamical measurements from Standing et al. 2023).*

### HYPOTHETICAL
* **1 Earth-Mass Desert Planet**: Completely invented for visual and habitability demonstration. Do not confuse this with the actual TOI-1338 b (Saturn-mass) or BEBOP-1c (Gas Giant).
* **1.10 AU Orbit**: Chosen to place the planet securely in the circumbinary habitable zone ($S \approx 1.5 - 1.7 S_\oplus$).
* **16° Obliquity**: Chosen for the day/night and seasonal visual effect.
* **Rotation Period**: Scaled for rapid visual diurnal cycles.
* **Procedural Desert Terrain**: Fictional landscape for the surface camera.

### MODELING APPROXIMATIONS
* **L $\propto$ M^4 Luminosity Scaling**: Used as an interactive main-sequence mass-luminosity approximation during the mass perturbation experiment, not as a strict observed luminosity law for TOI-1338 B.
* **Atmospheric Visual Effects**: Rayleigh and Mie scattering models to simulate atmospheric dust.
* **Real-Time Stability Diagnostic**: A real-time heuristic diagnostic based on current orbital state and Holman & Wiegert (1999) critical semi-major axis. It is NOT a proof of million-year N-body stability.
* **Integration**: Point-mass Newtonian three-body model with a symplectic Velocity Verlet (KDK) integrator and adaptive close-approach sub-stepping.

### INTERACTIVE EXPERIMENT
* **Star B Mass Perturbation**: Dynamically alter Star B's gravitational mass ($M/N$ keys) to investigate the resulting gravitational response (resonance pumping, eccentricity excitation, close encounters). The binary barycenter and velocities are dynamically rebalanced to preserve the center-of-mass frame and avoid artificial bulk momentum drift. The physical radius and effective temperature of Star B remain fixed to scientifically isolate the mass effect, while its approximated luminosity updates.

## Requirements & Installation

- Python 3.10 to 3.12 (Python 3.12 recommended)
- Taichi (`pip install taichi`)
- NumPy (`pip install numpy`)

## How to Run

Run with `uv`:
```bash
uv run --python 3.12 --with taichi sim.py
```
Or directly with Python 3.12:
```bash
py -3.12 sim.py
```

## Controls

| Key / Input | Action |
| :--- | :--- |
| **`C`** | Toggle Camera View (**Surface Desert View** $\longleftrightarrow$ **Star System Orbit**) |
| **`SPACE`** | Pause / Resume simulation |
| **`M` / `N`** | Increase / Decrease Star B Mass by $\pm 0.05\,M_\odot$ (Resonance experiment) |
| **`Q` / `E`** | Slow down / Speed up simulation time warp |
| **`R`** | Reset simulation to initial TOI-1338 baseline parameters |
| **Mouse Left Drag** | Look around horizon/sky (Surface View) or Orbit camera (System View) |
| **`W` / `S`** or **$\uparrow$ / $\downarrow$** | Zoom in / Zoom out (System View) |
