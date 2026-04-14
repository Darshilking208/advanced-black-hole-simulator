#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          🌌  3D BLACK HOLE SIMULATION  🌌                        ║
║          Physics-Based Python Visualization                      ║
║                                                                  ║
║  Features:                                                       ║
║   • Schwarzschild black hole with event horizon                  ║
║   • Differentially rotating accretion disk (temperature-colored) ║
║   • Keplerian orbital mechanics with Paczyński-Wiita correction   ║
║   • Particle trails, glow rings, photon sphere                   ║
║   • Twinkling star field background                              ║
║   • Full keyboard & mouse interaction                            ║
║                                                                  ║
║  Controls:                                                       ║
║   SPACE  – Pause / Resume                                        ║
║   R      – Reset all particles                                   ║
║   +/-    – Increase / Decrease black hole mass                   ║
║   Mouse  – Rotate 3-D view                                       ║
║   Scroll – Zoom in / out                                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ─────────────────────────────────────────────
import sys

# ── Third-party ───────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3-D projection)
    from matplotlib.patches import FancyArrowPatch
except ImportError as _e:
    print(f"\n✗  Missing dependency: {_e}")
    print("   Install everything with:\n")
    print("       pip install numpy matplotlib\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
#  SIMULATION CONSTANTS  (all normalised – no SI units)
# ══════════════════════════════════════════════════════════════════

G  = 1.0          # Gravitational constant  (normalised)
C  = 10.0         # Speed of light          (normalised)

# Black hole default parameters
DEFAULT_BH_MASS     = 100.0
DEFAULT_NUM_PARTICLES = 90
NUM_STARS           = 250
TRAIL_LENGTH        = 20          # frames of particle trail kept
DT                  = 0.006       # integration time-step
ANIMATION_INTERVAL  = 18          # ms between frames  (~55 fps)


# ══════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════

def schwarzschild_radius(mass: float) -> float:
    """
    r_s = 2 G M / c²
    The radius at which the escape velocity equals c.
    Anything inside r_s cannot escape — this is the event horizon.
    """
    return 2.0 * G * mass / (C ** 2)


def keplerian_speed(mass: float, r: float) -> float:
    """
    v_k = sqrt( G M / r )
    Circular orbital speed from Newtonian gravity.
    Used to give particles stable initial orbits.
    """
    return np.sqrt(G * mass / max(r, 1e-6))


def pseudo_newtonian_factor(r: float, rs: float) -> float:
    """
    Paczyński-Wiita pseudo-Newtonian correction factor.

    F = 1 / (1 - rs/r)²

    This cheap scalar tweak to Newtonian gravity reproduces several
    key features of Schwarzschild geometry:
      • Innermost Stable Circular Orbit (ISCO) at r = 3 rs
      • Unbounded force as r → rs  (particles spiral in)

    Capped at 10 to keep the integrator stable.
    """
    if r <= rs:
        return 10.0
    return min(10.0, 1.0 / (1.0 - rs / r) ** 2)


# ══════════════════════════════════════════════════════════════════
#  CLASS: BlackHole
# ══════════════════════════════════════════════════════════════════

class BlackHole:
    """
    The central singularity.

    Attributes
    ----------
    mass                : gravitational mass (normalised)
    schwarzschild_radius: event-horizon radius
    position            : always at the origin
    """

    def __init__(self, mass: float = DEFAULT_BH_MASS):
        self.mass     = mass
        self.position = np.zeros(3)
        self.schwarzschild_radius = schwarzschild_radius(mass)

    # ── Public API ───────────────────────────────────────────────

    def set_mass(self, mass: float):
        """Update mass and recompute event-horizon radius."""
        self.mass = max(10.0, mass)
        self.schwarzschild_radius = schwarzschild_radius(self.mass)

    def gravitational_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """
        a = G M / r_soft² * r̂   with Paczyński-Wiita correction.

        A small softening ε prevents the denominator ever hitting zero.
        """
        r_vec     = self.position - pos
        r         = np.linalg.norm(r_vec)
        if r < 1e-6:
            return np.zeros(3)

        softening = self.schwarzschild_radius * 0.05
        r_soft    = np.sqrt(r ** 2 + softening ** 2)

        # Base Newtonian acceleration
        acc = (G * self.mass / r_soft ** 3) * r_vec

        # Pseudo-Newtonian enhancement near the event horizon
        pn = pseudo_newtonian_factor(r, self.schwarzschild_radius)
        return acc * pn

    def captures(self, pos: np.ndarray) -> bool:
        """Return True if pos is inside the event horizon."""
        return np.linalg.norm(pos - self.position) < self.schwarzschild_radius * 0.95


# ══════════════════════════════════════════════════════════════════
#  CLASS: Particle
# ══════════════════════════════════════════════════════════════════

class Particle:
    """
    A test particle (star fragment, gas cloud, etc.) subject to
    gravitational acceleration.  Uses velocity-Verlet integration
    for accurate energy conservation in circular orbits.

    Attributes
    ----------
    position : current 3-D position
    velocity : current 3-D velocity
    color    : matplotlib colour string or hex
    size     : scatter point size
    active   : False once swallowed by the black hole
    trail    : deque of recent positions (for the glowing tail)
    """

    COLORS = ['#00ffff', '#ffffff', '#ffff44', '#ff9933',
              '#aaddff', '#ff6644', '#88ff88', '#ff88cc']

    def __init__(self, position: np.ndarray, velocity: np.ndarray,
                 color: str | None = None, size: float = 5.0):
        self.position  = position.astype(float)
        self.velocity  = velocity.astype(float)
        self.color     = color or np.random.choice(self.COLORS)
        self.size      = size
        self.active    = True
        self.trail: list[np.ndarray] = []

    def update(self, bh: BlackHole, dt: float):
        """Advance one time step using velocity-Verlet integration."""
        if not self.active:
            return

        # ── Velocity Verlet ───────────────────────────────────────
        # 1. Compute acceleration at current position
        acc = bh.gravitational_acceleration(self.position)

        # 2. Update velocity (half-kick → full-kick style is fine for
        #    the accuracy we need here)
        self.velocity += acc * dt

        # 3. Update position
        self.position += self.velocity * dt

        # ── Trail management ──────────────────────────────────────
        self.trail.append(self.position.copy())
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

        # ── Capture check ─────────────────────────────────────────
        if bh.captures(self.position):
            self.active = False


# ══════════════════════════════════════════════════════════════════
#  CLASS: AccretionDisk
# ══════════════════════════════════════════════════════════════════

class AccretionDisk:
    """
    Geometrically thin, optically thick accretion disk.

    Physics highlights
    ------------------
    • Temperature profile  T(r) ∝ r^(-3/4)   (Shakura-Sunyaev)
      → hotter (white/blue) near the ISCO, cooler (red) at the rim.
    • Differential (Keplerian) rotation: inner annuli lap outer ones.
      Ω(r) = sqrt( G M / r³ )
    • Slight vertical warp gives a 3-D feel.

    The disk is drawn as a dense scatter of coloured points so that
    matplotlib's depth sorting still looks reasonable.
    """

    def __init__(self, bh: BlackHole,
                 num_rings: int = 22, points_per_ring: int = 70):
        self.bh              = bh
        self.num_rings       = num_rings
        self.points_per_ring = points_per_ring
        self.rotation_angle  = 0.0      # master rotation accumulator

    # ── Internal helpers ─────────────────────────────────────────

    @property
    def inner_r(self) -> float:
        """ISCO ≈ 3 r_s in Paczyński-Wiita gravity."""
        return self.bh.schwarzschild_radius * 3.0

    @property
    def outer_r(self) -> float:
        return self.bh.schwarzschild_radius * 13.0

    @staticmethod
    def _temperature_color(temp: float) -> tuple:
        """
        Map normalised temperature [0,1] to an RGBA colour.

        temp ≈ 1 → innermost, hottest → bright white / blue-white
        temp ≈ 0 → outermost, coolest → deep red / dark orange
        """
        if temp > 0.80:
            return (1.0,  1.0,  min(1.0, 0.6 + temp * 0.5),  0.92)   # white-blue
        if temp > 0.55:
            return (1.0,  0.55 + temp * 0.45, 0.05,           0.85)   # yellow
        if temp > 0.30:
            return (1.0,  0.20 + temp * 0.45, 0.0,            0.75)   # orange
        return     (0.75, 0.05,                0.0,            0.65)   # deep red

    # ── Public API ────────────────────────────────────────────────

    def update(self, dt: float):
        """Advance the disk rotation by one time step."""
        # Angular velocity of the inner edge sets the overall pacing
        omega_inner = keplerian_speed(self.bh.mass, self.inner_r) / self.inner_r
        self.rotation_angle += omega_inner * dt * 0.35   # 0.35 = visual speed scale

    def get_points(self):
        """
        Return (x, y, z, colors, sizes) arrays for scatter-plotting
        the current disk snapshot.
        """
        radii = np.linspace(self.inner_r, self.outer_r, self.num_rings)
        angles = np.linspace(0, 2 * np.pi, self.points_per_ring, endpoint=False)

        xs, ys, zs, cols, szs = [], [], [], [], []

        for r in radii:
            # Normalised temperature: 1 at inner edge, 0 at outer edge
            temp = (self.outer_r - r) / (self.outer_r - self.inner_r)

            # Keplerian differential rotation offset
            omega = keplerian_speed(self.bh.mass, r) / r
            phi_offset = self.rotation_angle * omega / (
                keplerian_speed(self.bh.mass, self.inner_r) / self.inner_r
            )

            phi = angles + phi_offset

            # Gentle vertical warp: z oscillates with 2× the azimuthal frequency
            z_warp = 0.04 * r * np.sin(2 * phi + self.rotation_angle * 0.3)

            xs.extend(r * np.cos(phi))
            ys.extend(r * np.sin(phi))
            zs.extend(z_warp)

            color = self._temperature_color(temp)
            point_size = 2.5 + temp * 12.0
            cols.extend([color] * self.points_per_ring)
            szs.extend([point_size] * self.points_per_ring)

        return (np.array(xs), np.array(ys), np.array(zs),
                cols, np.array(szs))


# ══════════════════════════════════════════════════════════════════
#  CLASS: StarField
# ══════════════════════════════════════════════════════════════════

class StarField:
    """
    Static background of distant stars distributed on a large sphere.
    Each star has an independent twinkle phase so brightness flickers
    naturally over time.
    """

    def __init__(self, count: int = NUM_STARS, radius: float = 32.0):
        rng   = np.random.default_rng(seed=42)          # reproducible sky
        theta = rng.uniform(0,      np.pi,     count)
        phi   = rng.uniform(0, 2 * np.pi,      count)
        r     = rng.uniform(radius * 0.85, radius, count)

        self.x = r * np.sin(theta) * np.cos(phi)
        self.y = r * np.sin(theta) * np.sin(phi)
        self.z = r * np.cos(theta)

        self.base_sizes   = rng.uniform(0.4, 3.5, count)
        self.brightness   = rng.uniform(0.3, 1.0, count)
        self.twinkle_freq = rng.uniform(0.02, 0.08, count)
        self.twinkle_phase = rng.uniform(0, 2 * np.pi, count)

    def get_scatter_data(self, frame: int):
        """Return sizes and grayscale colours with current twinkle state."""
        twinkle = 0.5 + 0.5 * np.sin(self.twinkle_phase + frame * self.twinkle_freq)
        sizes   = self.base_sizes * (0.6 + 0.4 * twinkle)
        colors  = [(b * t, b * t, min(1.0, b * t + 0.05))
                   for b, t in zip(self.brightness, twinkle)]
        return sizes, colors


# ══════════════════════════════════════════════════════════════════
#  CLASS: BlackHoleSimulation  (orchestrator)
# ══════════════════════════════════════════════════════════════════

class BlackHoleSimulation:
    """
    Top-level class that owns all physics objects, the matplotlib
    figure, and the FuncAnimation loop.

    Architecture
    ────────────
    __init__         – create objects, build figure
    run()            – start FuncAnimation (blocking)
    _update()        – called each frame: physics step + redraw
    _draw_*()        – modular drawing helpers
    _on_key_press()  – keyboard handler
    """

    def __init__(self, bh_mass: float = DEFAULT_BH_MASS,
                 num_particles: int = DEFAULT_NUM_PARTICLES):
        # ── Physics objects ───────────────────────────────────────
        self.bh              = BlackHole(mass=bh_mass)
        self.disk            = AccretionDisk(self.bh)
        self.stars           = StarField()
        self.particles: list[Particle] = []
        self.num_particles   = num_particles

        # ── Simulation state ──────────────────────────────────────
        self.frame   = 0
        self.paused  = False
        self.rng     = np.random.default_rng()

        # ── Spawn initial particles ───────────────────────────────
        for _ in range(self.num_particles):
            self.particles.append(self._make_particle())

        # ── Build figure ──────────────────────────────────────────
        self._build_figure()

    # ── Figure construction ───────────────────────────────────────

    def _build_figure(self):
        """Create the matplotlib figure, 3-D axes, and overlay text."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(13, 10), facecolor='#000005')
        self.ax  = self.fig.add_subplot(111, projection='3d',
                                        facecolor='#000005')
        self.ax.set_axis_off()

        # ── Static overlay text ───────────────────────────────────
        self.fig.text(
            0.5, 0.965,
            '🌌  3D Black Hole Simulation',
            ha='center', va='top',
            fontsize=17, color='#e8e8ff',
            fontfamily='monospace', fontweight='bold'
        )
        self.fig.text(
            0.5, 0.025,
            'SPACE · pause/resume    R · reset    +/- · mass    '
            'Mouse · rotate    Scroll · zoom',
            ha='center', fontsize=9, color='#555577',
            fontfamily='monospace'
        )

        # ── Dynamic info panel (top-left) ─────────────────────────
        self.info_text = self.fig.text(
            0.015, 0.96, '',
            va='top', fontsize=8.5,
            color='#44ddff', fontfamily='monospace'
        )

        # ── Key binding ───────────────────────────────────────────
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    # ── Keyboard handler ──────────────────────────────────────────

    def _on_key_press(self, event):
        """
        Supported keys
        ──────────────
        SPACE  toggle pause
        r      reset all particles
        +      increase black hole mass by 20 %
        -      decrease black hole mass by 20 %
        """
        if event.key == ' ':
            self.paused = not self.paused

        elif event.key == 'r':
            self.particles = [self._make_particle()
                              for _ in range(self.num_particles)]

        elif event.key == '+':
            self.bh.set_mass(self.bh.mass * 1.20)
            self.disk = AccretionDisk(self.bh)

        elif event.key == '-':
            self.bh.set_mass(self.bh.mass * 0.80)
            self.disk = AccretionDisk(self.bh)

    # ── Particle factory ──────────────────────────────────────────

    def _make_particle(self, spawn_outer: bool = False) -> Particle:
        """
        Create one particle on a nearly-circular Keplerian orbit.

        If spawn_outer=True the particle starts farther out (used
        when replacing captured particles so the disk refills slowly).

        Orbital mechanics summary
        ─────────────────────────
        1. Pick a random radius r in [r_ISCO, r_outer].
        2. Compute the Keplerian circular speed v_k = sqrt(G M / r).
        3. Tilt the orbit plane slightly off the equator (|θ| < 10°).
        4. Construct the velocity vector perpendicular to the position
           vector, guaranteeing (near-)circular initial motion.
        5. Add a small random radial kick to create mild eccentricity.
        """
        rs     = self.bh.schwarzschild_radius
        r_min  = rs * (8.0 if spawn_outer else 3.2)
        r_max  = rs * 15.0

        r   = self.rng.uniform(r_min, r_max)
        phi = self.rng.uniform(0, 2 * np.pi)
        inc = self.rng.uniform(-0.17, 0.17)   # orbital inclination (≈ ±10°)

        pos = np.array([
            r * np.cos(phi) * np.cos(inc),
            r * np.sin(phi) * np.cos(inc),
            r * np.sin(inc)
        ])

        # Circular speed with slight randomisation (0.88–1.02 × v_k)
        v_k    = keplerian_speed(self.bh.mass, r) * self.rng.uniform(0.88, 1.02)
        v_radial = self.rng.uniform(-0.04, 0.04) * v_k   # tiny radial kick

        vel = np.array([
            -v_k * np.sin(phi) + v_radial * np.cos(phi) * np.cos(inc),
             v_k * np.cos(phi) + v_radial * np.sin(phi) * np.cos(inc),
             v_radial * np.sin(inc)
        ])

        return Particle(
            position = pos,
            velocity = vel,
            size     = self.rng.uniform(3.5, 9.0)
        )

    # ── Drawing helpers ───────────────────────────────────────────

    def _set_axis_limits(self):
        rs  = self.bh.schwarzschild_radius
        lim = rs * 16.5
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim * 0.38, lim * 0.38)

    def _draw_event_horizon(self):
        """
        Render the event horizon as a perfectly black sphere.
        Uses a surface mesh.  Everything inside r_s is hidden inside.
        """
        rs   = self.bh.schwarzschild_radius
        u    = np.linspace(0, 2 * np.pi, 32)
        v    = np.linspace(0,     np.pi, 22)
        x    = rs * np.outer(np.cos(u), np.sin(v))
        y    = rs * np.outer(np.sin(u), np.sin(v))
        z    = rs * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_surface(x, y, z, color='#000000', alpha=1.0,
                              linewidth=0, zorder=10)

    def _draw_photon_sphere(self):
        """
        Photon sphere at r = 1.5 r_s — the radius where photons orbit.
        Drawn as a faint dashed ring in the equatorial plane.
        """
        rs    = self.bh.schwarzschild_radius
        r_ph  = 1.5 * rs
        theta = np.linspace(0, 2 * np.pi, 120)
        self.ax.plot(r_ph * np.cos(theta),
                     r_ph * np.sin(theta),
                     np.zeros_like(theta),
                     color='#ffee00', alpha=0.22,
                     linewidth=1.0, linestyle='--')

    def _draw_glow_rings(self):
        """
        Semi-transparent concentric rings that simulate the bright
        photon-capture glow visible around real black holes.
        Inner rings are warm orange; outer rings fade away.
        """
        rs = self.bh.schwarzschild_radius
        glow_specs = [
            (rs * 1.10, '#ff9900', 0.70, 2.5),
            (rs * 1.35, '#ff6600', 0.45, 1.8),
            (rs * 1.65, '#cc3300', 0.25, 1.2),
            (rs * 2.10, '#881100', 0.12, 0.8),
        ]
        theta = np.linspace(0, 2 * np.pi, 180)
        for r, color, alpha, lw in glow_specs:
            self.ax.plot(r * np.cos(theta),
                         r * np.sin(theta),
                         np.zeros_like(theta),
                         color=color, alpha=alpha, linewidth=lw)

    def _draw_stars(self):
        """Render the twinkling star field."""
        szs, cols = self.stars.get_scatter_data(self.frame)
        self.ax.scatter(self.stars.x, self.stars.y, self.stars.z,
                        s=szs, c=cols, alpha=0.85, zorder=1,
                        depthshade=False)

    def _draw_disk(self):
        """Render the accretion disk scatter points."""
        dx, dy, dz, cols, szs = self.disk.get_points()
        self.ax.scatter(dx, dy, dz, c=cols, s=szs,
                        alpha=0.75, zorder=3, depthshade=False)

    def _draw_particles(self):
        """Render active particles and their glowing trails."""
        active = [p for p in self.particles if p.active]
        if not active:
            return

        # ── Scatter plot (current positions) ─────────────────────
        px     = [p.position[0] for p in active]
        py     = [p.position[1] for p in active]
        pz     = [p.position[2] for p in active]
        pcols  = [p.color       for p in active]
        psizes = [p.size        for p in active]
        self.ax.scatter(px, py, pz, c=pcols, s=psizes,
                        alpha=0.95, zorder=8, depthshade=False)

        # ── Trail lines (fading towards the tail) ─────────────────
        for p in active:
            n = len(p.trail)
            if n < 3:
                continue
            trail = np.asarray(p.trail)
            # Draw trail in segments with increasing opacity
            step = max(1, n // 10)
            for i in range(step, n, step):
                alpha = (i / n) ** 1.8 * 0.55   # gamma curve → bright tip
                self.ax.plot(
                    trail[i-step:i+1, 0],
                    trail[i-step:i+1, 1],
                    trail[i-step:i+1, 2],
                    color=p.color, alpha=alpha,
                    linewidth=0.7, zorder=7
                )

    def _draw_gravitational_lensing_hint(self):
        """
        Simple visual hint of gravitational lensing: concentric
        faint ellipses behind the black hole that warp slightly,
        imitating the Einstein ring for background stars.
        """
        rs = self.bh.schwarzschild_radius
        theta = np.linspace(0, 2 * np.pi, 200)
        # Vary the warp with simulation time for a subtle shimmer
        warp = 0.12 * np.sin(theta * 3 + self.frame * 0.02)
        for scale, alpha in [(2.8, 0.10), (3.6, 0.07), (4.5, 0.05)]:
            r = rs * scale * (1 + warp * 0.15)
            self.ax.plot(r * np.cos(theta), r * np.sin(theta),
                         rs * 0.05 * np.sin(theta * 2),
                         color='#aaccff', alpha=alpha, linewidth=0.8)

    # ── Main animation callback ───────────────────────────────────

    def _update(self, _frame_ignored: int):
        """
        Called by FuncAnimation each frame.

        Order of operations
        ───────────────────
        1. Clear axes and re-set limits / style.
        2. Physics step (if not paused):
           a. rotate accretion disk
           b. integrate each particle
           c. replace captured particles
        3. Draw layers (back → front):
           star field → lensing hint → disk → glow rings →
           photon sphere → event horizon → particles
        4. Update info overlay.
        """
        # ── Reset axes ────────────────────────────────────────────
        self.ax.cla()
        self.ax.set_facecolor('#000005')
        self.ax.set_axis_off()
        self._set_axis_limits()

        # ── Physics step ──────────────────────────────────────────
        if not self.paused:
            self.disk.update(DT)

            for p in self.particles:
                p.update(self.bh, DT)

            # Replace captured particles (spawn from outer edge
            # so the viewer sees them spiral in over time)
            n_captured = sum(1 for p in self.particles if not p.active)
            self.particles = [p for p in self.particles if p.active]
            for _ in range(n_captured):
                self.particles.append(self._make_particle(spawn_outer=True))

            self.frame += 1

        # ── Draw layers ───────────────────────────────────────────
        self._draw_stars()
        self._draw_gravitational_lensing_hint()
        self._draw_disk()
        self._draw_glow_rings()
        self._draw_photon_sphere()
        self._draw_event_horizon()
        self._draw_particles()

        # ── Info overlay ──────────────────────────────────────────
        rs     = self.bh.schwarzschild_radius
        active = sum(1 for p in self.particles if p.active)
        status = "⏸ PAUSED" if self.paused else "▶ RUNNING"

        self.info_text.set_text(
            f"Frame   : {self.frame:06d}\n"
            f"Status  : {status}\n"
            f"Particles: {active}/{self.num_particles}\n"
            f"BH Mass : {self.bh.mass:7.1f}\n"
            f"r_s     : {rs:7.3f}\n"
            f"ISCO    : {3*rs:7.3f}\n"
            f"[+/-] mass  [r] reset  [SPACE] pause"
        )

        return []

    # ── Public entry point ────────────────────────────────────────

    def run(self):
        """Launch the real-time animation (blocking call)."""
        print("\n" + "═" * 60)
        print("  🌌  3D BLACK HOLE SIMULATION  — Python / matplotlib")
        print("═" * 60)
        print(f"  Black hole mass  : {self.bh.mass}")
        print(f"  Schwarzschild r_s: {self.bh.schwarzschild_radius:.4f}")
        print(f"  ISCO radius      : {3*self.bh.schwarzschild_radius:.4f}")
        print(f"  Particles        : {self.num_particles}")
        print(f"  Stars            : {NUM_STARS}")
        print()
        print("  Controls")
        print("  ─────────────────────────────────────────")
        print("  SPACE      pause / resume")
        print("  R          reset all particles")
        print("  + / -      increase / decrease BH mass")
        print("  Mouse drag rotate the view")
        print("  Scroll     zoom in / out")
        print("  Window ✕   exit")
        print("═" * 60 + "\n")

        self._ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=None,                  # infinite
            interval=ANIMATION_INTERVAL,  # ms
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Dependency versions ───────────────────────────────────────
    print(f"  numpy      {np.__version__}")
    print(f"  matplotlib {matplotlib.__version__}")
    print()

    # ── Optional: allow mass & particle count from CLI ────────────
    import argparse

    parser = argparse.ArgumentParser(
        description="3D Black Hole Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mass',      type=float, default=DEFAULT_BH_MASS,
                        help='Black hole mass (normalised)')
    parser.add_argument('--particles', type=int,   default=DEFAULT_NUM_PARTICLES,
                        help='Number of orbiting particles')
    args = parser.parse_args()

    # ── Run ───────────────────────────────────────────────────────
    sim = BlackHoleSimulation(bh_mass=args.mass, num_particles=args.particles)
    sim.run()
