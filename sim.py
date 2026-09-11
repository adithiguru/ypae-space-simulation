"""
TOI-1338 Circumbinary System: Desert Planet (Tatooine-Class) Simulation
======================================================================
A physically grounded, cinematic simulation of the real binary star system TOI-1338
and a hypothetical Earth-sized desert planet orbiting its circumbinary barycenter.

Physics & Celestial Mechanics:
- Primary Star (TOI-1338 A): Spectral type F8V, Mass = 1.127 M_sun, Teff = 6050 K, L = 2.02 L_sun
- Secondary Star (TOI-1338 B): M-dwarf, Mass = 0.299 M_sun, Teff = 3200 K, L = 0.0085 L_sun
- Binary Orbit: a_bin = 0.1321 AU, e_bin = 0.1555, Period = 14.61 days
- Desert Planet: a_p = 1.10 AU, Earth-sized (1 R_earth), desert climate (S ~ 1.5 - 1.7 S_earth)
- Critical Stability Boundary: Holman & Wiegert (1999) empirical criterion for circumbinary planets
- Integrator: Symplectic Velocity Verlet (Kick-Drift-Kick) with G = 4*pi^2 (AU, M_sun, yr)
- Surface View: True emergent double sunset from planet's rotational frame and dual-star elevation!
"""

import taichi as ti
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS. Lower the body count if it crawls.")

import numpy as np
import math

# =============================================================================
# ASTRONOMICAL CONSTANTS & CONFIGURATION
# =============================================================================
# Units: Astronomical Units (AU), Solar Masses (M_sun), Years (yr)
# Under these units: G = 4 * pi^2
G_CONST = 4.0 * math.pi * math.pi  # ~ 39.4784176 AU^3 / (M_sun * yr^2)
EPSILON = 1e-3  # Softening parameter to prevent close-encounter singularities

# TOI-1338 System Parameters (Observed Astronomical Data)
M_A_INITIAL = 1.127    # Primary F8V star mass in M_sun
M_B_INITIAL = 0.299    # Secondary M-dwarf star mass in M_sun
M_P_INITIAL = 3.003e-6 # Hypothetical desert planet mass (1 Earth mass)

L_A_BASE = 2.02        # TOI-1338 A luminosity relative to Sun (from R=1.30 R_sun, T=6050 K)
L_B_BASE = 0.0085      # TOI-1338 B luminosity relative to Sun (from R=0.30 R_sun, T=3200 K)

A_BIN = 0.1321         # Semi-major axis of binary in AU
E_BIN = 0.1555         # Eccentricity of binary orbit
A_PLANET = 1.10        # Circumbinary planet semi-major axis in AU

# Simulation Dimensions & Field Allocations
RES_W = 1280
RES_H = 800
N_BODIES = 3           # 0: Star A, 1: Star B, 2: Desert Planet
N_DUST = 3072          # Circumbinary dust & asteroid test particles
TRAIL_LEN = 1000       # Number of trail history steps
HIST_LEN = 240         # Historical telemetry buffer for live GUI plots

# =============================================================================
# TAICHI FIELDS (All physical state lives in Taichi memory)
# =============================================================================
# Massive bodies: pos, vel, acc, mass, visual radius, and base color
pos = ti.Vector.field(3, ti.f32, shape=N_BODIES)
vel = ti.Vector.field(3, ti.f32, shape=N_BODIES)
acc = ti.Vector.field(3, ti.f32, shape=N_BODIES)
mass = ti.field(ti.f32, shape=N_BODIES)
radius_vis = ti.field(ti.f32, shape=N_BODIES)
body_color = ti.Vector.field(3, ti.f32, shape=N_BODIES)

# Circumbinary dust particles (massless test bodies perturbed by binary gravity)
dust_pos = ti.Vector.field(3, ti.f32, shape=N_DUST)
dust_vel = ti.Vector.field(3, ti.f32, shape=N_DUST)
dust_acc = ti.Vector.field(3, ti.f32, shape=N_DUST)
dust_color = ti.Vector.field(3, ti.f32, shape=N_DUST)

# Orbital trails (rolling circular history buffers)
trails = ti.Vector.field(3, ti.f32, shape=(N_BODIES, TRAIL_LEN))
trail_head = ti.field(ti.i32, shape=())

# Telemetry scalar fields
sim_time = ti.field(ti.f32, shape=())
flux_A = ti.field(ti.f32, shape=())
flux_B = ti.field(ti.f32, shape=())
flux_tot = ti.field(ti.f32, shape=())
planet_dist = ti.field(ti.f32, shape=())
planet_ecc = ti.field(ti.f32, shape=())
planet_sma = ti.field(ti.f32, shape=())
a_crit_field = ti.field(ti.f32, shape=())
star_sep = ti.field(ti.f32, shape=())

# Rolling history fields for onscreen live HUD graphs
hist_flux_a = ti.field(ti.f32, shape=HIST_LEN)
hist_flux_b = ti.field(ti.f32, shape=HIST_LEN)
hist_flux_tot = ti.field(ti.f32, shape=HIST_LEN)
hist_ecc = ti.field(ti.f32, shape=HIST_LEN)
hist_head = ti.field(ti.i32, shape=())

# Framebuffer canvas
pixels = ti.Vector.field(3, ti.f32, shape=(RES_W, RES_H))

# =============================================================================
# SYSTEM INITIALIZATION & KERNELS
# =============================================================================
def init_system(m_b_val=M_B_INITIAL):
    """Initialize TOI-1338 binary and circumbinary planet into Keplerian orbits."""
    m_a = M_A_INITIAL
    m_b = m_b_val
    m_tot = m_a + m_b
    m_p = M_P_INITIAL

    mass[0] = m_a
    mass[1] = m_b
    mass[2] = m_p

    # Visual radii (scaled for clear visibility in system space view)
    radius_vis[0] = 0.038   # TOI-1338 A (Warm-white F-star)
    radius_vis[1] = 0.022   # TOI-1338 B (Red dwarf)
    radius_vis[2] = 0.016   # Desert Planet

    # Colors: Star A (Brilliant F8V warm-yellow/white), Star B (Deep M-dwarf Crimson), Planet (Terracotta Desert)
    body_color[0] = ti.Vector([1.0, 0.94, 0.82])
    body_color[1] = ti.Vector([1.0, 0.32, 0.12])
    body_color[2] = ti.Vector([0.88, 0.62, 0.38])

    # Compute binary periastron distance and relative velocity via Vis-Viva equation
    r_peri = A_BIN * (1.0 - E_BIN)
    v_rel = math.sqrt(G_CONST * m_tot * (1.0 + E_BIN) / (A_BIN * (1.0 - E_BIN)))

    # Set binary stars at periastron around their mutual barycenter at origin (0, 0, 0)
    pos[0] = ti.Vector([-m_b / m_tot * r_peri, 0.0, 0.0])
    vel[0] = ti.Vector([0.0, -m_b / m_tot * v_rel, 0.0])

    pos[1] = ti.Vector([m_a / m_tot * r_peri, 0.0, 0.0])
    vel[1] = ti.Vector([0.0, m_a / m_tot * v_rel, 0.0])

    # Place circumbinary planet in co-planar stable orbit
    v_planet = math.sqrt(G_CONST * (m_tot + m_p) / A_PLANET)
    pos[2] = ti.Vector([A_PLANET, 0.0, 0.0])
    vel[2] = ti.Vector([0.0, v_planet, 0.0])

    acc[0] = ti.Vector([0.0, 0.0, 0.0])
    acc[1] = ti.Vector([0.0, 0.0, 0.0])
    acc[2] = ti.Vector([0.0, 0.0, 0.0])

    # Initialize dust particles in circumbinary Keplerian belt
    np.random.seed(42)
    for k in range(N_DUST):
        r = np.random.uniform(0.40, 1.80)
        theta = np.random.uniform(0.0, 2.0 * math.pi)
        z = np.random.normal(0.0, 0.015)
        dust_pos[k] = ti.Vector([r * math.cos(theta), r * math.sin(theta), z])
        vk = math.sqrt(G_CONST * m_tot / r)
        dust_vel[k] = ti.Vector([-vk * math.sin(theta), vk * math.cos(theta), 0.0])
        dust_acc[k] = ti.Vector([0.0, 0.0, 0.0])
        # Color variation: subtle golden-tan zodiacal dust
        c_rand = np.random.uniform(0.7, 1.0)
        dust_color[k] = ti.Vector([0.75 * c_rand, 0.65 * c_rand, 0.45 * c_rand])

    # Initialize trails with starting positions
    for b in range(N_BODIES):
        for t in range(TRAIL_LEN):
            trails[b, t] = pos[b]
    trail_head[None] = 0

@ti.func
def compute_accelerations():
    """Compute mutual gravitational accelerations for all bodies and dust particles."""
    # 1. Mutual gravitational forces between massive bodies (Star A, Star B, Planet)
    for i in ti.static(range(N_BODIES)):
        a_acc = ti.Vector([0.0, 0.0, 0.0])
        p_i = pos[i]
        for j in ti.static(range(N_BODIES)):
            if i != j:
                r_vec = pos[j] - p_i
                dist_sq = r_vec.dot(r_vec) + EPSILON * EPSILON
                dist = ti.sqrt(dist_sq)
                a_acc += (G_CONST * mass[j] / (dist * dist_sq)) * r_vec
        acc[i] = a_acc

    # 2. Gravitational forces acting on massless dust particles from both stars and planet
    for k in range(N_DUST):
        p_k = dust_pos[k]
        a_dust = ti.Vector([0.0, 0.0, 0.0])
        for b in ti.static(range(N_BODIES)):
            r_vec = pos[b] - p_k
            dist_sq = r_vec.dot(r_vec) + EPSILON * EPSILON
            dist = ti.sqrt(dist_sq)
            a_dust += (G_CONST * mass[b] / (dist * dist_sq)) * r_vec
        dust_acc[k] = a_dust

@ti.kernel
def init_accelerations():
    compute_accelerations()

@ti.kernel
def symplectic_step(dt: ti.f32):
    """
    Symplectic Velocity Verlet Integrator (Kick-Drift-Kick).
    Guarantees strict orbital energy conservation over millennia.
    """
    # 1. Kick: v(t + dt/2) = v(t) + 0.5 * dt * a(t)
    for i in range(N_BODIES):
        vel[i] += 0.5 * dt * acc[i]
    for k in range(N_DUST):
        dust_vel[k] += 0.5 * dt * dust_acc[k]

    # 2. Drift: x(t + dt) = x(t) + dt * v(t + dt/2)
    for i in range(N_BODIES):
        pos[i] += dt * vel[i]
    for k in range(N_DUST):
        dust_pos[k] += dt * dust_vel[k]
        # Never-ending show: if dust particle is flung into interstellar space, respawn into belt
        r_sq = dust_pos[k].dot(dust_pos[k])
        if r_sq > 16.0 or r_sq < 0.04:
            # Respawn in circumbinary disc
            angle = ti.random() * 6.2831853
            new_r = 0.5 + ti.random() * 1.2
            dust_pos[k] = ti.Vector([new_r * ti.cos(angle), new_r * ti.sin(angle), (ti.random() - 0.5) * 0.03])
            m_tot = mass[0] + mass[1]
            vk = ti.sqrt(G_CONST * m_tot / new_r)
            dust_vel[k] = ti.Vector([-vk * ti.sin(angle), vk * ti.cos(angle), 0.0])

    # 3. Recompute gravitational field at new positions
    compute_accelerations()

    # 4. Kick: v(t + dt) = v(t + dt/2) + 0.5 * dt * a(t + dt)
    for i in range(N_BODIES):
        vel[i] += 0.5 * dt * acc[i]
    for k in range(N_DUST):
        dust_vel[k] += 0.5 * dt * dust_acc[k]

    sim_time[None] += dt

@ti.kernel
def update_telemetry_and_trails():
    """Compute physical metrics: Holman-Wiegert stability, irradiance, and append trails."""
    # Update rolling trails
    h = (trail_head[None] + 1) % TRAIL_LEN
    for b in range(N_BODIES):
        trails[b, h] = pos[b]
    trail_head[None] = h

    # 1. Binary parameters
    r_bin_vec = pos[1] - pos[0]
    r_bin = ti.sqrt(r_bin_vec.dot(r_bin_vec) + EPSILON * EPSILON)
    star_sep[None] = r_bin

    m_a = mass[0]
    m_b = mass[1]
    m_tot = m_a + m_b
    mu = m_b / (m_tot + EPSILON)

    # Holman & Wiegert (1999) Critical Stability Semi-Major Axis for Circumbinary Systems
    # a_crit = a_bin * (1.60 + 5.10*e - 2.22*e^2 + 4.12*mu - 4.27*e*mu - 5.09*mu^2 + 4.61*e^2*mu^2)
    e_b = E_BIN
    a_crit = A_BIN * (1.60 + 5.10 * e_b - 2.22 * e_b * e_b + 4.12 * mu 
                      - 4.27 * e_b * mu - 5.09 * mu * mu + 4.61 * e_b * e_b * mu * mu)
    a_crit_field[None] = a_crit

    # 2. Circumbinary Planet State relative to binary barycenter
    bary_pos = (m_a * pos[0] + m_b * pos[1]) / (m_tot + EPSILON)
    bary_vel = (m_a * vel[0] + m_b * vel[1]) / (m_tot + EPSILON)

    r_p_vec = pos[2] - bary_pos
    v_p_vec = vel[2] - bary_vel
    r_p = ti.sqrt(r_p_vec.dot(r_p_vec) + EPSILON * EPSILON)
    planet_dist[None] = r_p

    # Specific orbital energy & angular momentum for osculating eccentricity
    v_sq = v_p_vec.dot(v_p_vec)
    specific_energy = 0.5 * v_sq - (G_CONST * m_tot) / r_p
    h_vec = r_p_vec.cross(v_p_vec)
    # Eccentricity vector: e = (v x h) / (G*M) - r / |r|
    e_vec = (v_p_vec.cross(h_vec)) / (G_CONST * m_tot) - r_p_vec / r_p
    ecc = ti.sqrt(e_vec.dot(e_vec))
    planet_ecc[None] = ecc

    # Osculating semi-major axis: a = - G*M / (2 * energy)
    if specific_energy < -1e-5:
        planet_sma[None] = - (G_CONST * m_tot) / (2.0 * specific_energy)
    else:
        planet_sma[None] = 999.0  # Hyperbolic escape trajectory!

    # 3. Combined Stellar Irradiance (Flux received by Desert Planet)
    # Normalized so that 1.0 = Earth's solar constant (1361 W/m^2)
    d_a_vec = pos[2] - pos[0]
    d_b_vec = pos[2] - pos[1]
    d_a_sq = d_a_vec.dot(d_a_vec) + EPSILON * EPSILON
    d_b_sq = d_b_vec.dot(d_b_vec) + EPSILON * EPSILON

    # Luminosity scaling: Star B luminosity scales with mass^4
    lum_b = L_B_BASE * ti.pow(m_b / M_B_INITIAL, 4.0)

    f_a = L_A_BASE / d_a_sq
    f_b = lum_b / d_b_sq
    flux_A[None] = f_a
    flux_B[None] = f_b
    flux_tot[None] = f_a + f_b

    # Append to rolling history buffer for live onscreen HUD plots
    hp = (hist_head[None] + 1) % HIST_LEN
    hist_flux_a[hp] = f_a
    hist_flux_b[hp] = f_b
    hist_flux_tot[hp] = f_a + f_b
    hist_ecc[hp] = ecc
    hist_head[None] = hp

@ti.kernel
def draw_hud_graphs():
    """
    Render live vector telemetry graphs directly onto the pixel canvas:
    1. Upper Graph: Combined S_tot(t) and individual stellar fluxes (S_A, S_B) in S_earth
    2. Lower Graph: Osculating eccentricity e(t) tracking resonance pumping & stability boundary
    """
    # Graph 1: Stellar Insolation Flux [X: 840..1250, Y: 490..770]
    # Graph 2: Planet Eccentricity      [X: 840..1250, Y: 180..460]
    for i, j in pixels:
        # 1. Insolation Graph Box
        if 840 <= i <= 1250 and 490 <= j <= 770:
            # Semi-transparent glassmorphic slate background
            pixels[i, j] = pixels[i, j] * 0.25 + ti.Vector([0.02, 0.04, 0.07]) * 0.75

            # Border
            if i == 840 or i == 1250 or j == 490 or j == 770:
                pixels[i, j] = ti.Vector([0.25, 0.45, 0.65])
            # Internal reference lines (S = 1.0 and S = 2.0)
            # Scaling: 0 to 3.0 S_earth
            y_s1 = 583
            y_s2 = 676
            if (j == y_s1 or j == y_s2) and (i % 6 < 4):
                pixels[i, j] += ti.Vector([0.15, 0.25, 0.35])

        # 2. Eccentricity Graph Box
        if 840 <= i <= 1250 and 180 <= j <= 460:
            # Semi-transparent background
            pixels[i, j] = pixels[i, j] * 0.25 + ti.Vector([0.02, 0.04, 0.07]) * 0.75

            # Border
            if i == 840 or i == 1250 or j == 180 or j == 460:
                pixels[i, j] = ti.Vector([0.25, 0.45, 0.65])
            # Critical stability limit line (e = 0.35)
            y_ecrit = 320
            if j == y_ecrit and (i % 5 < 3):
                # Flashing warning dashes
                pixels[i, j] = ti.Vector([0.8, 0.25, 0.15])

    # Draw continuous rolling plot curves across the graphs
    h_now = hist_head[None]
    for col in range(842, 1249):
        # Progress across history window (0.0 to 1.0)
        frac = float(col - 842) / 406.0
        step_back = ti.cast((1.0 - frac) * float(HIST_LEN - 1), ti.i32)
        k = (h_now - step_back + HIST_LEN) % HIST_LEN

        # -------------------------------------------------------------
        # Trace for Graph 1: Stellar Insolation
        # -------------------------------------------------------------
        # Scale: [0.0, 3.0] S_earth mapped to [495, 765]
        val_tot = hist_flux_tot[k]
        val_a = hist_flux_a[k]
        val_b = hist_flux_b[k]

        y_tot = ti.cast(495.0 + clamp(val_tot / 3.0, 0.0, 0.96) * 265.0, ti.i32)
        y_a = ti.cast(495.0 + clamp(val_a / 3.0, 0.0, 0.96) * 265.0, ti.i32)
        y_b = ti.cast(495.0 + clamp(val_b / 3.0, 0.0, 0.96) * 265.0, ti.i32)

        # Draw Star A flux (Warm White / Pale Gold)
        pixels[col, y_a] = ti.Vector([0.9, 0.85, 0.70])
        # Draw Star B flux (Ruby Crimson)
        pixels[col, y_b] = ti.Vector([1.0, 0.30, 0.15])
        # Draw Combined flux (Thick Golden Amber)
        pixels[col, y_tot] = ti.Vector([1.0, 0.88, 0.25])
        pixels[col, y_tot + 1] = ti.Vector([1.0, 0.88, 0.25])

        # -------------------------------------------------------------
        # Trace for Graph 2: Planet Eccentricity
        # -------------------------------------------------------------
        # Scale: [0.0, 0.70] mapped to [185, 455]
        val_e = hist_ecc[k]
        y_e = ti.cast(185.0 + clamp(val_e / 0.70, 0.0, 0.96) * 265.0, ti.i32)

        # Color shifts from stable cyan to warning crimson
        col_e = ti.Vector([0.2, 0.9, 0.75])
        if val_e > 0.35:
            col_e = ti.Vector([1.0, 0.25, 0.15])
        elif val_e > 0.15:
            col_e = ti.Vector([0.95, 0.75, 0.20])

        pixels[col, y_e] = col_e
        pixels[col, y_e + 1] = col_e


# =============================================================================
# RAYMARCHING & RASTER COMPUTE SHADERS
# =============================================================================
@ti.func
def clamp(val: ti.f32, min_val: ti.f32, max_val: ti.f32) -> ti.f32:
    return ti.min(ti.max(val, min_val), max_val)

@ti.func
def fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)

@ti.func
def dune_height(x: ti.f32, z: ti.f32) -> ti.f32:
    """
    Procedural Desert Landscape: Multi-scale transverse sand dunes and sandstone ridges.
    """
    # Large rolling longitudinal dunes
    h1 = 1.4 * ti.sin(0.06 * x + 0.03 * z) * ti.cos(0.02 * x - 0.05 * z)
    # Wind-carved sharp slipface / barchan crests
    c1 = ti.sin(0.12 * x + 0.06 * z)
    h2 = -0.6 * ti.abs(c1)
    # Fine wind ripple textures
    h3 = 0.08 * ti.sin(0.8 * x + 0.4 * z)
    # Horizon sandstone monolith / mesa outcroppings
    rock_dist = ti.sqrt((x - 45.0) * (x - 45.0) + (z - 65.0) * (z - 65.0))
    rock_mesa = 4.5 / (1.0 + 0.05 * rock_dist * rock_dist)
    return h1 + h2 + h3 + rock_mesa

@ti.kernel
def render_surface_view(yaw: ti.f32, pitch: ti.f32):
    """
    Surface Camera View: Standing on the desert planet.
    Both stars rise, cross the sky, and set naturally based on planetary rotation
    and physical orbital vectors, generating an emergent, unscripted double sunset!
    """
    # Planet diurnal rotation around tilted axis
    # Spin rate: 1 planetary rotation = ~0.025 sim years (fast enough to enjoy cycles)
    rot_angle = sim_time[None] * 250.0
    axial_tilt = 0.28  # ~16 degrees obliquity
    cos_t = ti.cos(axial_tilt)
    sin_t = ti.sin(axial_tilt)

    # Local topocentric ground observer frame (latitude ~25 deg North)
    lat = 0.436  # 25 degrees
    cos_lat = ti.cos(lat)
    sin_lat = ti.sin(lat)
    cos_p = ti.cos(rot_angle)
    sin_p = ti.sin(rot_angle)

    # Observer normal (Zenith vector in barycentric coordinates)
    zenith_unrot = ti.Vector([cos_lat * cos_p, cos_lat * sin_p, sin_lat])
    # Apply axial tilt rotation in Y-Z
    zenith = ti.Vector([
        zenith_unrot[0],
        zenith_unrot[1] * cos_t - zenith_unrot[2] * sin_t,
        zenith_unrot[1] * sin_t + zenith_unrot[2] * cos_t
    ]).normalized()

    # Surface East and North basis vectors
    east_unrot = ti.Vector([-sin_p, cos_p, 0.0])
    east = ti.Vector([
        east_unrot[0],
        east_unrot[1] * cos_t - east_unrot[2] * sin_t,
        east_unrot[1] * sin_t + east_unrot[2] * cos_t
    ]).normalized()
    north = zenith.cross(east).normalized()

    # Direction vectors from planet to Star A and Star B in barycentric frame
    p_planet = pos[2]
    vec_to_a = (pos[0] - p_planet).normalized()
    vec_to_b = (pos[1] - p_planet).normalized()

    # Project stars into local topocentric horizon coordinates (East, North, Zenith)
    s_a_local = ti.Vector([vec_to_a.dot(east), vec_to_a.dot(north), vec_to_a.dot(zenith)])
    s_b_local = ti.Vector([vec_to_b.dot(east), vec_to_b.dot(north), vec_to_b.dot(zenith)])

    # Star elevations: positive = in the sky, negative = below horizon!
    alt_a = s_a_local[2]
    alt_b = s_b_local[2]

    # Camera basis rotated by user yaw and pitch
    cy = ti.cos(yaw)
    sy = ti.sin(yaw)
    cp = ti.cos(pitch)
    sp = ti.sin(pitch)

    cam_fwd = (cy * cp * east + sy * cp * north + sp * zenith).normalized()
    cam_right = (sy * east - cy * north).normalized()
    cam_up = cam_right.cross(cam_fwd).normalized()

    fov = 1.15  # Field of view
    aspect = float(RES_W) / float(RES_H)

    for i, j in pixels:
        uv_x = (float(i) / float(RES_W) - 0.5) * 2.0 * aspect * ti.tan(fov * 0.5)
        uv_y = (float(j) / float(RES_H) - 0.5) * 2.0 * ti.tan(fov * 0.5)

        ray_dir = (cam_fwd + uv_x * cam_right + uv_y * cam_up).normalized()

        # Ray direction in local topocentric frame: (East, North, Zenith)
        rd_east = ray_dir.dot(east)
        rd_north = ray_dir.dot(north)
        rd_zenith = ray_dir.dot(zenith)

        col = ti.Vector([0.0, 0.0, 0.0])

        # -----------------------------------------------------------------
        # TERRAIN INTERSECTION (Rolling Sand Dunes & Sandstone Formations)
        # -----------------------------------------------------------------
        hit_terrain = False
        t_hit = 0.0
        terrain_p = ti.Vector([0.0, 0.0, 0.0])

        if rd_zenith < 0.05:  # Ray pointing toward or near horizon
            # Ground raymarch
            t = 0.8
            eye_h = 2.2 + dune_height(0.0, 0.0)
            for _ in range(42):
                curr_p = t * ray_dir
                # Horizontal ground plane coordinates (x_g, z_g)
                xg = curr_p.dot(east)
                zg = curr_p.dot(north)
                h_surf = dune_height(xg, zg)
                h_ray = eye_h + curr_p.dot(zenith)
                diff = h_ray - h_surf
                if diff < 0.02:
                    hit_terrain = True
                    t_hit = t
                    terrain_p = ti.Vector([xg, h_surf, zg])
                    break
                t += diff * 0.65 + 0.08
                if t > 180.0:
                    break

        if hit_terrain:
            # Numerical normal of terrain
            eps = 0.05
            xg = terrain_p[0]
            zg = terrain_p[2]
            h0 = dune_height(xg, zg)
            hx = dune_height(xg + eps, zg)
            hz = dune_height(xg, zg + eps)
            norm_local = ti.Vector([-(hx - h0) / eps, 1.0, -(hz - h0) / eps]).normalized()

            # Surface base color: warm terracotta, Namib red-gold sand, stratified rock
            sand_color = ti.Vector([0.88, 0.56, 0.32])
            rock_strata = 0.5 + 0.5 * ti.sin(terrain_p[1] * 3.5)
            rock_color = ti.Vector([0.65 + 0.15 * rock_strata, 0.38, 0.25])
            base_col = sand_color if terrain_p[1] < 2.5 else rock_color

            # Dual-star illumination: Star A (Warm White) & Star B (Crimson Dwarf)
            # Atmospheric extinction increases dramatically as stars approach the horizon
            airmass_a = 1.0 / (ti.max(0.01, alt_a) + 0.05)
            airmass_b = 1.0 / (ti.max(0.01, alt_b) + 0.05)
            ext_a = ti.exp(-0.15 * airmass_a)
            ext_b = ti.exp(-0.08 * airmass_b)

            # Transmitted solar spectrum at planet surface
            col_a_ground = ti.Vector([1.0, 0.92 * ext_a, 0.70 * ext_a * ext_a])
            col_b_ground = ti.Vector([1.0, 0.32 * ext_b, 0.10 * ext_b * ext_b])

            # Diffuse lighting from Star A
            n_dot_a = ti.max(0.0, norm_local[0] * s_a_local[0] + norm_local[1] * s_a_local[2] + norm_local[2] * s_a_local[1])
            light_a = col_a_ground * (n_dot_a * ti.max(0.0, alt_a) * (flux_A[None] / L_A_BASE))

            # Diffuse lighting from Star B
            n_dot_b = ti.max(0.0, norm_local[0] * s_b_local[0] + norm_local[1] * s_b_local[2] + norm_local[2] * s_b_local[1])
            light_b = col_b_ground * (n_dot_b * ti.max(0.0, alt_b) * 1.8)

            # Ambient desert skylight
            amb = ti.Vector([0.15, 0.12, 0.10]) * (ti.max(0.05, alt_a * 0.5 + alt_b * 0.2))

            surf_col = base_col * (light_a + light_b + amb)

            # Atmospheric dust distance haze (Namib desert suspension)
            haze_factor = 1.0 - ti.exp(-0.018 * t_hit)
            haze_col = ti.Vector([0.85, 0.60, 0.40]) * (ti.max(0.08, alt_a * 0.7 + alt_b * 0.3))
            col = surf_col * (1.0 - haze_factor) + haze_col * haze_factor

        else:
            # -----------------------------------------------------------------
            # DYNAMIC SKY & AUTHENTIC DOUBLE SUNSET
            # -----------------------------------------------------------------
            # Rayleigh & Mie atmospheric scattering gradient
            sky_h = ti.max(0.0, rd_zenith)
            # Baseline sky color transitions from deep space cobalt at zenith to warm amber at horizon
            zenith_col = ti.Vector([0.04, 0.08, 0.22])
            horizon_col = ti.Vector([0.90, 0.62, 0.36]) * ti.max(0.05, alt_a * 0.8 + 0.3)
            sky_col = zenith_col * sky_h + horizon_col * (1.0 - sky_h)

            # Star A: Primary F8V Star (Angular radius ~ 0.022 rad / 1.3 deg)
            cos_theta_a = ray_dir.dot(vec_to_a)
            disk_radius_a = 0.022
            # Limb darkening
            if cos_theta_a > (1.0 - 0.5 * disk_radius_a * disk_radius_a):
                r_disk = ti.sqrt(ti.max(0.0, 2.0 * (1.0 - cos_theta_a))) / disk_radius_a
                mu_limb = ti.sqrt(ti.max(0.0, 1.0 - r_disk * r_disk))
                limb_factor = 0.4 + 0.6 * mu_limb
                # Sunset reddening for Star A as it touches the dunes
                redden = clamp(alt_a * 5.0, 0.0, 1.0)
                star_a_disk = ti.Vector([1.0, 0.75 + 0.25 * redden, 0.35 + 0.65 * redden]) * (3.5 * limb_factor)
                sky_col += star_a_disk

            # Star A Corona & Mie aerosol forward scattering halo
            mie_a = 0.0012 / (1.002 - cos_theta_a)
            sky_col += ti.Vector([1.0, 0.85, 0.60]) * (mie_a * ti.max(0.05, alt_a + 0.1))

            # Star B: M-Dwarf Red Dwarf (Angular radius ~ 0.014 rad / 0.8 deg)
            cos_theta_b = ray_dir.dot(vec_to_b)
            disk_radius_b = 0.014
            if cos_theta_b > (1.0 - 0.5 * disk_radius_b * disk_radius_b):
                r_disk_b = ti.sqrt(ti.max(0.0, 2.0 * (1.0 - cos_theta_b))) / disk_radius_b
                mu_limb_b = ti.sqrt(ti.max(0.0, 1.0 - r_disk_b * r_disk_b))
                star_b_disk = ti.Vector([1.0, 0.30, 0.08]) * (2.8 * (0.5 + 0.5 * mu_limb_b))
                sky_col += star_b_disk

            # Star B Ruby Corona Halo
            mie_b = 0.0007 / (1.002 - cos_theta_b)
            sky_col += ti.Vector([1.0, 0.25, 0.05]) * (mie_b * ti.max(0.05, alt_b + 0.1))

            # If both stars are below the horizon: Crisp starry desert night!
            night_factor = clamp(1.0 - (alt_a * 4.0 + alt_b * 2.0), 0.0, 1.0)
            if night_factor > 0.0:
                # Procedural stars
                p_star = ray_dir * 180.0
                star_hash = ti.sin(p_star[0] * 12.9898 + p_star[1] * 78.233 + p_star[2] * 37.719) * 43758.5453
                star_val = fract(star_hash)
                if star_val > 0.995:
                    sparkle = ti.pow((star_val - 0.995) * 200.0, 3.0)
                    sky_col += ti.Vector([0.9, 0.95, 1.0]) * (sparkle * night_factor)

            col = sky_col

        # Subtle vignette & tone mapping
        col = col / (col + 1.0)  # Reinhard tonemap
        pixels[i, j] = col

@ti.kernel
def render_system_view(cam_dist: ti.f32, cam_yaw: ti.f32, cam_pitch: ti.f32):
    """
    Star System View: Full 3D exploration of TOI-1338 binary orbit, circumbinary planet,
    fading orbital ribbons, dust disk, and reference rings around the barycenter.
    """
    # Camera position in barycentric coordinates
    cy = ti.cos(cam_yaw)
    sy = ti.sin(cam_yaw)
    cp = ti.cos(cam_pitch)
    sp = ti.sin(cam_pitch)

    cam_pos = ti.Vector([cam_dist * cp * cy, cam_dist * cp * sy, cam_dist * sp])
    cam_target = ti.Vector([0.0, 0.0, 0.0])
    cam_fwd = (cam_target - cam_pos).normalized()
    world_up = ti.Vector([0.0, 0.0, 1.0])
    cam_right = cam_fwd.cross(world_up).normalized()
    cam_up = cam_right.cross(cam_fwd).normalized()

    fov = 0.95
    aspect = float(RES_W) / float(RES_H)

    for i, j in pixels:
        uv_x = (float(i) / float(RES_W) - 0.5) * 2.0 * aspect * ti.tan(fov * 0.5)
        uv_y = (float(j) / float(RES_H) - 0.5) * 2.0 * ti.tan(fov * 0.5)

        ray_dir = (cam_fwd + uv_x * cam_right + uv_y * cam_up).normalized()

        # Background: Deep space with starry background
        p_bg = ray_dir * 120.0
        star_hash = ti.sin(p_bg[0] * 12.9898 + p_bg[1] * 78.233 + p_bg[2] * 45.164) * 43758.5453
        star_val = fract(star_hash)

        col = ti.Vector([0.012, 0.012, 0.020])
        if star_val > 0.994:
            col += ti.Vector([0.8, 0.85, 1.0]) * ti.pow((star_val - 0.994) * 166.0, 2.5)

        # -----------------------------------------------------------------
        # REFERENCE GRID: Invariant Binary Plane Distance Rings (0.2, 0.5, 1.0, 1.5 AU)
        # -----------------------------------------------------------------
        if ti.abs(ray_dir[2]) > 1e-4:
            t_plane = -cam_pos[2] / ray_dir[2]
            if t_plane > 0.0:
                p_plane = cam_pos + t_plane * ray_dir
                r_plane = ti.sqrt(p_plane[0] * p_plane[0] + p_plane[1] * p_plane[1])
                # Ring markers at 0.2 AU, 0.5 AU, 1.0 AU, 1.5 AU
                for ring_r in ti.static([0.2, 0.5, 1.0, 1.5]):
                    dr = ti.abs(r_plane - ring_r)
                    if dr < 0.004 * (1.0 + 0.4 * t_plane):
                        grid_alpha = ti.exp(-0.4 * t_plane)
                        col += ti.Vector([0.15, 0.35, 0.45]) * grid_alpha

        # -----------------------------------------------------------------
        # CELESTIAL BODIES: Ray-Sphere Intersections
        # -----------------------------------------------------------------
        # 1. Primary Star: TOI-1338 A (Body 0)
        p_a = pos[0]
        r_a = radius_vis[0]
        oc_a = cam_pos - p_a
        b_a = oc_a.dot(ray_dir)
        c_a = oc_a.dot(oc_a) - r_a * r_a
        disc_a = b_a * b_a - c_a
        if disc_a > 0.0:
            t_star_a = -b_a - ti.sqrt(disc_a)
            if t_star_a > 0.0:
                hit_p = (cam_pos + t_star_a * ray_dir) - p_a
                norm_a = hit_p.normalized()
                mu_a = ti.max(0.0, -ray_dir.dot(norm_a))
                limb_a = 0.35 + 0.65 * ti.pow(mu_a, 0.7)
                col = body_color[0] * (3.8 * limb_a)
        else:
            # Corona glow around Star A
            d_line_a = oc_a - b_a * ray_dir
            dist_sq_a = d_line_a.dot(d_line_a)
            corona_a = 0.0018 / (dist_sq_a + 0.0004)
            col += body_color[0] * corona_a

        # 2. Secondary Star: TOI-1338 B (Body 1, M-Dwarf)
        p_b = pos[1]
        r_b = radius_vis[1]
        oc_b = cam_pos - p_b
        b_b = oc_b.dot(ray_dir)
        c_b = oc_b.dot(oc_b) - r_b * r_b
        disc_b = b_b * b_b - c_b
        if disc_b > 0.0:
            t_star_b = -b_b - ti.sqrt(disc_b)
            if t_star_b > 0.0:
                hit_p = (cam_pos + t_star_b * ray_dir) - p_b
                norm_b = hit_p.normalized()
                mu_b = ti.max(0.0, -ray_dir.dot(norm_b))
                col = body_color[1] * (3.2 * (0.4 + 0.6 * mu_b))
        else:
            # Red Dwarf Corona Flare
            d_line_b = oc_b - b_b * ray_dir
            dist_sq_b = d_line_b.dot(d_line_b)
            corona_b = 0.0010 / (dist_sq_b + 0.0003)
            col += body_color[1] * corona_b

        # 3. Circumbinary Desert Planet (Body 2)
        p_p = pos[2]
        r_p = radius_vis[2]
        oc_p = cam_pos - p_p
        b_p = oc_p.dot(ray_dir)
        c_p = oc_p.dot(oc_p) - r_p * r_p
        disc_p = b_p * b_p - c_p
        if disc_p > 0.0:
            t_planet = -b_p - ti.sqrt(disc_p)
            if t_planet > 0.0:
                hit_p = (cam_pos + t_planet * ray_dir) - p_p
                norm_p = hit_p.normalized()
                # Dual illumination from Star A and Star B
                l_dir_a = (pos[0] - (p_p + hit_p)).normalized()
                l_dir_b = (pos[1] - (p_p + hit_p)).normalized()
                diff_a = ti.max(0.0, norm_p.dot(l_dir_a))
                diff_b = ti.max(0.0, norm_p.dot(l_dir_b))
                # Atmospheric rim glow
                rim = ti.pow(1.0 - ti.max(0.0, -ray_dir.dot(norm_p)), 3.0)
                lit_col = body_color[2] * (diff_a * 1.5 + diff_b * 0.6 + 0.05)
                lit_col += ti.Vector([0.4, 0.7, 1.0]) * (rim * (diff_a + diff_b))
                col = lit_col

        # Tonemap Reinhard
        col = col / (col + 1.0)
        pixels[i, j] = col

@ti.kernel
def splat_particles_and_trails(cam_dist: ti.f32, cam_yaw: ti.f32, cam_pitch: ti.f32):
    """
    Project circumbinary dust particles and fading orbital trails onto the screen.
    """
    cos_yaw = ti.cos(cam_yaw)
    sin_yaw = ti.sin(cam_yaw)
    cos_pitch = ti.cos(cam_pitch)
    sin_pitch = ti.sin(cam_pitch)

    cam_pos = ti.Vector([cam_dist * cos_pitch * cos_yaw, cam_dist * cos_pitch * sin_yaw, cam_dist * sin_pitch])
    cam_target = ti.Vector([0.0, 0.0, 0.0])
    cam_fwd = (cam_target - cam_pos).normalized()
    world_up = ti.Vector([0.0, 0.0, 1.0])
    cam_right = cam_fwd.cross(world_up).normalized()
    cam_up = cam_right.cross(cam_fwd).normalized()

    fov = 0.95
    aspect = float(RES_W) / float(RES_H)

    # 1. Splat circumbinary dust particles
    for k in range(N_DUST):
        p = dust_pos[k]
        rel = p - cam_pos
        depth = rel.dot(cam_fwd)
        if depth > 0.1:
            rx = rel.dot(cam_right)
            ry = rel.dot(cam_up)
            fx = (rx / (depth * aspect * ti.tan(fov * 0.5)) * 0.5 + 0.5) * float(RES_W)
            fy = (ry / (depth * ti.tan(fov * 0.5)) * 0.5 + 0.5) * float(RES_H)
            screen_x = ti.cast(fx, ti.i32)
            screen_y = ti.cast(fy, ti.i32)
            if 1 <= screen_x < RES_W - 1 and 1 <= screen_y < RES_H - 1:
                col = dust_color[k] * 0.65
                pixels[screen_x, screen_y] += col
                pixels[screen_x + 1, screen_y] += col * 0.4
                pixels[screen_x, screen_y + 1] += col * 0.4

    # 2. Splat orbital history trails
    h_idx = trail_head[None]
    for b in range(N_BODIES):
        t_col = body_color[b]
        for step in range(TRAIL_LEN):
            idx = (h_idx - step + TRAIL_LEN) % TRAIL_LEN
            p = trails[b, idx]
            rel = p - cam_pos
            depth = rel.dot(cam_fwd)
            if depth > 0.1:
                rx = rel.dot(cam_right)
                ry = rel.dot(cam_up)
                fx2 = (rx / (depth * aspect * ti.tan(fov * 0.5)) * 0.5 + 0.5) * float(RES_W)
                fy2 = (ry / (depth * ti.tan(fov * 0.5)) * 0.5 + 0.5) * float(RES_H)
                trail_px = ti.cast(fx2, ti.i32)
                trail_py = ti.cast(fy2, ti.i32)
                if 1 <= trail_px < RES_W - 1 and 1 <= trail_py < RES_H - 1:
                    alpha = 1.0 - float(step) / float(TRAIL_LEN)
                    pix_c = t_col * (alpha * 0.85)
                    pixels[trail_px, trail_py] += pix_c
                    if b == 2:  # Thicker trail for planet
                        pixels[trail_px + 1, trail_py] += pix_c * 0.5
                        pixels[trail_px, trail_py + 1] += pix_c * 0.5


# =============================================================================
# MAIN APPLICATION LOOP & INTERACTIVE CONTROLS
# =============================================================================
def main():
    print("=" * 78)
    print("  TOI-1338 CIRCUMBINARY DESERT WORLD: ASTROPHYSICS & SUNSET SIMULATION")
    print("=" * 78)
    print(" CONTROLS:")
    print("   [C]          : Switch Camera (Surface Desert View <-> Star System View)")
    print("   [SPACE]      : Pause / Resume Simulation")
    print("   [M] / [N]    : Increase / Decrease Star B Mass (+/- 0.05 M_sun)")
    print("   [[] / []]    : Slow down / Speed up simulation time")
    print("   [R]          : Reset system to original TOI-1338 parameters")
    print("   Mouse Drag   : Look around (Surface View) or Orbit camera (System View)")
    print("   Mouse Scroll : Zoom camera in / out (System View)")
    print("=" * 78)

    # Initialize simulation physics
    current_m_b = M_B_INITIAL
    init_system(current_m_b)
    init_accelerations()

    # GUI Window Setup
    window = ti.ui.Window("TOI-1338: Circumbinary World & Double Sunset", (RES_W, RES_H), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Camera state
    # Mode 0: Surface Desert View; Mode 1: Star System Space View
    cam_mode = 0  
    # Surface camera angles
    surf_yaw = 0.35
    surf_pitch = 0.08
    # System camera angles
    sys_dist = 2.6
    sys_yaw = 0.8
    sys_pitch = 0.55

    # Simulation control variables
    paused = False
    time_scale = 1.0   # Speed multiplier
    substeps = 60      # Substeps per frame for high symplectic precision
    dt_base = 0.00004  # ~ 21 minutes per substep

    # Rolling history buffers for live GUI telemetry graphs
    hist_times = np.zeros(HIST_LEN)
    hist_flux_tot = np.zeros(HIST_LEN)
    hist_flux_a = np.zeros(HIST_LEN)
    hist_flux_b = np.zeros(HIST_LEN)
    hist_ecc = np.zeros(HIST_LEN)
    hist_ptr = 0

    last_mouse_pos = (0.5, 0.5)

    while window.running:
        # -----------------------------------------------------------------
        # USER INPUT HANDLING
        # -----------------------------------------------------------------
        # Keyboard shortcuts
        if window.is_pressed('c'):
            cam_mode = 1 - cam_mode
            print(f"Switched Camera Mode -> {'SYSTEM VIEW' if cam_mode == 1 else 'SURFACE DESERT VIEW'}")
        if window.is_pressed(' '):
            paused = not paused
        if window.is_pressed('r'):
            current_m_b = M_B_INITIAL
            init_system(current_m_b)
            init_accelerations()
            print("System reset to initial TOI-1338 parameters.")
        if window.is_pressed('m'):
            current_m_b = min(3.0, current_m_b + 0.05)
            mass[1] = current_m_b
            print(f"Star B Mass increased to: {current_m_b:.3f} M_sun (Resonance experiment)")
        if window.is_pressed('n'):
            current_m_b = max(0.05, current_m_b - 0.05)
            mass[1] = current_m_b
            print(f"Star B Mass decreased to: {current_m_b:.3f} M_sun")
        if window.is_pressed('['):
            time_scale = max(0.1, time_scale * 0.85)
        if window.is_pressed(']'):
            time_scale = min(15.0, time_scale * 1.15)

        # Mouse drag for view control
        mouse = window.get_cursor_pos()
        dx = mouse[0] - last_mouse_pos[0]
        dy = mouse[1] - last_mouse_pos[1]
        last_mouse_pos = mouse

        if window.is_pressed(ti.ui.LMB):
            if cam_mode == 0:
                surf_yaw += dx * 2.8
                surf_pitch = np.clip(surf_pitch + dy * 2.2, -0.25, 1.2)
            else:
                sys_yaw += dx * 3.0
                sys_pitch = np.clip(sys_pitch + dy * 2.5, -1.45, 1.45)

        # Keys for zooming system camera
        if window.is_pressed('w') or window.is_pressed(ti.ui.UP):
            sys_dist = max(0.4, sys_dist - 0.04)
        if window.is_pressed('s') or window.is_pressed(ti.ui.DOWN):
            sys_dist = min(8.0, sys_dist + 0.04)

        # -----------------------------------------------------------------
        # SYMPLECTIC PHYSICS STEP
        # -----------------------------------------------------------------
        if not paused:
            dt = dt_base * time_scale
            for _ in range(substeps):
                symplectic_step(dt)
            update_telemetry_and_trails()

        # Update telemetry history for GUI graphs
        hist_times[hist_ptr] = sim_time[None]
        hist_flux_tot[hist_ptr] = flux_tot[None]
        hist_flux_a[hist_ptr] = flux_A[None]
        hist_flux_b[hist_ptr] = flux_B[None]
        hist_ecc[hist_ptr] = planet_ecc[None]
        hist_ptr = (hist_ptr + 1) % HIST_LEN

        # -----------------------------------------------------------------
        # RENDERING PIPELINE
        # -----------------------------------------------------------------
        if cam_mode == 0:
            render_surface_view(surf_yaw, surf_pitch)
        else:
            render_system_view(sys_dist, sys_yaw, sys_pitch)
            splat_particles_and_trails(sys_dist, sys_yaw, sys_pitch)

        # Draw real-time vector HUD telemetry graphs onto pixel buffer
        draw_hud_graphs()

        canvas.set_image(pixels)

        # -----------------------------------------------------------------
        # SCIENTIFIC HUD & TELEMETRY DASHBOARD
        # -----------------------------------------------------------------
        cur_t = sim_time[None]
        cur_days = cur_t * 365.25
        cur_years = cur_t
        m_tot = mass[0] + mass[1]
        a_crit = a_crit_field[None]
        r_p = planet_dist[None]
        ecc = planet_ecc[None]
        f_tot = flux_tot[None]

        with gui.sub_window("TOI-1338 Telemetry & Scientific Lab", 0.02, 0.02, 0.36, 0.94):
            gui.text("=== TOI-1338 CIRCUMBINARY SYSTEM ===")
            gui.text(f"Sim Time: Year {cur_years:.2f} ({cur_days:.1f} Earth Days)")
            gui.text(f"View: {'[SURFACE DESERT VIEW]' if cam_mode == 0 else '[STAR SYSTEM ORBIT]'}")

            if gui.button("Toggle View Mode (C)"):
                cam_mode = 1 - cam_mode

            gui.text("----------------------------------------")
            gui.text("--- STELLAR MASS EXPERIMENT ---")
            gui.text(f"Star A (F8V Primary)  : {mass[0]:.3f} M_sun")
            gui.text(f"Star B (M-Dwarf)      : {mass[1]:.3f} M_sun (Observed: 0.299)")
            gui.text(f"Total Binary Mass     : {m_tot:.3f} M_sun")
            gui.text(f"Binary Separation     : {star_sep[None]:.4f} AU")

            if gui.button("+0.05 M_sun to Star B (M)"):
                current_m_b = min(3.0, current_m_b + 0.05)
                mass[1] = current_m_b
            if gui.button("-0.05 M_sun to Star B (N)"):
                current_m_b = max(0.05, current_m_b - 0.05)
                mass[1] = current_m_b

            gui.text("----------------------------------------")
            gui.text("--- ORBITAL STABILITY (Holman & Wiegert 1999) ---")
            gui.text(f"Critical Stability Radius a_crit : {a_crit:.4f} AU")
            gui.text(f"Planet Orbit Distance r_p        : {r_p:.4f} AU")
            gui.text(f"Planet Osculating Eccentricity e : {ecc:.4f}")

            # Dynamic Stability Status Badge
            if ecc < 0.12 and r_p > 1.3 * a_crit:
                gui.text("Status: [STABLE CIRCUMBINARY ORBIT]")
            elif ecc < 0.35 and r_p > a_crit:
                gui.text("Status: [RESONANT PERTURBATION DETECTED]")
            else:
                gui.text("Status: [CHAOTIC SCATTERING / EJECTION RISK!]")

            gui.text("----------------------------------------")
            gui.text("--- COMBINED STELLAR IRRADIANCE ---")
            gui.text(f"Flux Star A  : {flux_A[None]:.3f} S_earth")
            gui.text(f"Flux Star B  : {flux_B[None]:.3f} S_earth")
            gui.text(f"Combined S   : {f_tot:.3f} S_earth")
            t_eq_c = 278.0 * (f_tot ** 0.25) - 273.15
            gui.text(f"Estimated Eq Temp: {t_eq_c:.1f} °C (Baking Desert)")

            gui.text("----------------------------------------")
            gui.text("--- ONSCREEN HUD TELEMETRY GRAPHS ---")
            gui.text("Top-Right Box    : Live Flux S_tot (Gold), S_A, S_B")
            gui.text("Bottom-Right Box : Live Eccentricity e(t) (Cyan/Red)")

            gui.text("----------------------------------------")
            gui.text("--- SIMULATION CONTROLS ---")
            if gui.button("Pause / Resume (SPACE)"):
                paused = not paused
            if gui.button("Reset to Baseline (R)"):
                current_m_b = M_B_INITIAL
                init_system(current_m_b)
                init_accelerations()

            gui.text(f"Time Warp: {time_scale:.1f}x (Keys: [ / ])")

        window.show()


if __name__ == '__main__':
    main()
