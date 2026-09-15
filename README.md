# TOI-1338 Circumbinary System: Desert Planet & Double Sunset Simulation

A scientifically grounded, cinematic simulation of the real observed binary star system **TOI-1338** and a hypothetical Earth-sized desert planet orbiting its circumbinary barycenter, built in Python + Taichi.

## Features

- **Real Observed Binary (TOI-1338)**:
  - **TOI-1338 A**: F8V primary star ($1.127\,M_\odot$, $T_{\text{eff}} \approx 6050\,\text{K}$, $L \approx 2.02\,L_\odot$) with radiant warm-white light, limb darkening, and golden corona.
  - **TOI-1338 B**: M-dwarf companion ($0.299\,M_\odot$, $T_{\text{eff}} \approx 3200\,\text{K}$, $L \approx 0.0085\,L_\odot$) with deep ruby-crimson illumination and flares.
  - **Binary Orbit**: $a_{\text{bin}} = 0.1321\,\text{AU}$, $e_{\text{bin}} = 0.1555$, period $P = 14.61\,\text{days}$.
- **Emergent Double Sunset (Surface View)**:
  - View from the surface of the rotating desert world (latitude $25^\circ\text{ N}$, $16^\circ$ axial obliquity).
  - Raymarched dunes, wind ripples, sandstone monoliths, and dynamic Rayleigh & Mie dust atmospheric scattering.
  - Star A and Star B rise, traverse, and set naturally based on instantaneous line-of-sight vectors and planetary rotation, creating an authentic, unscripted double sunset.
- **Star System Exploration (Space View)**:
  - Full 3D exploration around the binary barycenter: 3D stars, lit desert planet with day/night terminator and atmospheric rim glow.
  - Fading multi-color orbital history ribbon trails (gold for Star A, crimson for Star B, cyan for the desert planet).
  - Circumbinary dust/asteroid swarm (3,072 particles) perturbed by the binary gravitational potential.
  - Concentric distance reference rings at $0.2$, $0.5$, $1.0$, and $1.5\,\text{AU}$.
- **Interactive Gravitational Mass Experiment**:
  - Dynamically alter Star B's mass ($M/N$ keys or GUI buttons) and observe the real gravitational response: resonance pumping, eccentricity excitation, close encounters, or interstellar ejection.
- **Procedural Deep-Space Skybox (Zero-Asset, Zero-VRAM)**:
  - Scientifically calibrated celestial sphere depicting the Milky Way and deep space as seen from the TOI-1338 system (~1,300 light-years away).
  - Tilted galactic disc plane with diffuse interstellar dust lanes and galactic core bulge glow.
  - Multi-frequency ionized emission nebulae: Hydrogen-Alpha ($656.3\,\text{nm}$ deep crimson) and Oxygen-III ($500.7\,\text{nm}$ teal/cyan).
  - Multi-spectral stellar classification: O/B blue giants, A/F white stars, G-class solar yellows, and M-class red dwarfs, with sub-pixel diffraction spikes on prominent stars.
  - Dynamic atmospheric extinction: seamlessly emerges above the desert dunes as the double suns set and day turns into night.
- **Scientific HUD & Live Telemetry Graphs**:
  - Real-time telemetry: simulation days/years, stellar masses, binary separation, orbital eccentricity, Holman & Wiegert (1999) critical stability radius $a_{\text{crit}}$, and combined stellar flux ($S/S_\oplus$).
  - Live vector graphs of combined & individual stellar insolation $S_{\text{tot}}(t)$ and planetary eccentricity $e(t)$.

## Approximations & Omissions

> *We approximated the gravitational interaction using a point-mass Newtonian three-body model with a symplectic Velocity Verlet integrator and empirical Holman–Wiegert stability boundary, scaling Star B's luminosity as $L \propto M^4$ during mass modification.*  
> *We left out general relativistic precession, tidal dissipation, and planetary atmospheric greenhouse feedbacks.*

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
