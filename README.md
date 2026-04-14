# 🌌 3D Black Hole Simulation

A physically-inspired, real-time 3D black hole simulation built entirely with Python, NumPy, and Matplotlib.

---

## 📦 Installation

```bash
pip install numpy matplotlib
```

> Python 3.10+ recommended (uses `str | None` union type hint).
> For Python 3.9 use: replace `str | None` with `Optional[str]` after adding `from typing import Optional`.

---

## 🚀 Running

```bash
# Default settings
python black_hole_simulation.py

# Custom mass and particle count
python black_hole_simulation.py --mass 200 --particles 120
```

---

## 🎮 Controls

| Key / Input | Action |
|-------------|--------|
| `SPACE` | Pause / Resume |
| `R` | Reset all particles |
| `+` | Increase black hole mass × 1.2 |
| `-` | Decrease black hole mass × 0.8 |
| Mouse drag | Rotate the 3D view |
| Scroll wheel | Zoom in / out |

---

## 🧱 Project Architecture

```
black_hole_simulation.py
│
├── class BlackHole          – Mass, Schwarzschild radius, gravity field
├── class Particle           – Position/velocity, Verlet integrator, trail
├── class AccretionDisk      – Differential rotation, temperature colouring
├── class StarField          – Background stars with twinkle effect
└── class BlackHoleSimulation – Orchestrator: figure, animation loop, drawing
```

---

## ⚛️ Physics Explained

### 1. Schwarzschild Radius
```
r_s = 2 G M / c²
```
The "point of no return". Anything crossing r_s cannot escape.  
In this simulation r_s is the radius of the black event-horizon sphere.

### 2. Gravitational Acceleration
```
a = G M / (r² + ε²)^(3/2) × r̂
```
Standard Newtonian inverse-square law with a tiny softening `ε` to prevent division by zero.

### 3. Paczyński–Wiita Pseudo-Newtonian Correction
```
F_PW = F_Newton / (1 − r_s/r)²
```
A scalar multiplier that reproduces key features of Schwarzschild geometry cheaply:
- Innermost Stable Circular Orbit (ISCO) at r = 3 r_s
- Diverging force as r → r_s (particles spiral in catastrophically)

### 4. Keplerian Orbits
```
v_k = √(G M / r)
```
Used to initialise particles on near-circular orbits. Small random perturbations create realistic eccentricity.

### 5. Accretion Disk Temperature Profile
```
T(r) ∝ r^(−3/4)    [Shakura–Sunyaev thin-disk model]
```
- Inner edge (hottest) → white / blue-white  
- Mid disk → yellow / orange  
- Outer edge (coolest) → deep red

### 6. Differential Rotation
```
Ω(r) = √(G M / r³)
```
Inner annuli of the disk complete more revolutions per second than outer ones — just like planets orbiting the Sun.

### 7. Velocity-Verlet Integration
```
v(t+dt) = v(t) + a(t) · dt
x(t+dt) = x(t) + v(t+dt) · dt
```
A simple symplectic integrator that conserves energy better than Euler's method, keeping circular orbits stable for thousands of frames.

---

## 🎨 Visual Features

| Feature | Implementation |
|---------|---------------|
| Event horizon | Black surface mesh sphere at r_s |
| Photon sphere | Dashed ring at 1.5 r_s |
| Glow rings | 4 concentric orange/red rings with alpha falloff |
| Accretion disk | ~1500 scatter points, temperature-coloured |
| Particle trails | Segment-by-segment with gamma-curve opacity |
| Lensing hint | Shimmering faint ellipses behind the BH |
| Star field | 250 stars with independent twinkle frequencies |

---

## 📈 Performance Tips

- Reduce `NUM_STARS` or `points_per_ring` in the constants if the animation is slow.
- Increase `ANIMATION_INTERVAL` (ms) to target a lower frame rate.
- On high-DPI displays, set `figsize=(10, 8)` for a smaller window.

---

## 📚 References

- Paczyński & Wiita (1980) — pseudo-Newtonian potential  
- Shakura & Sunyaev (1973) — thin accretion disk model  
- Misner, Thorne & Wheeler — *Gravitation* (1973)
