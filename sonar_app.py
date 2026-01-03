import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import time

# ==============================================================================
# PART 1: COMPATIBILITY & OPTIMIZATION
# ==============================================================================
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

st.set_page_config(layout="wide", page_title="SonarSim Pro (Strict)")

# ==============================================================================
# PART 2: IMMUTABLE PHYSICS ENGINE (PRESERVED)
# ==============================================================================

@njit(fastmath=False)
def get_bathymetry_offset_jit(y, mode_id, max_y):
    if mode_id == 0: return 0.0
    if mode_id == 1: return (y / max_y) * 5.0
    if mode_id == 2: return -3.0 * np.exp(-((y - max_y/2)**2) / 25.0)
    if mode_id == 3: return 3.0 * np.exp(-((y - max_y/2)**2) / 9.0)
    return 0.0

@njit(fastmath=False)
def get_bathymetry_slope_jit(y, mode_id, max_y):
    if mode_id == 0: return 0.0
    if mode_id == 1: return 5.0 / max_y
    if mode_id == 2:
        center = max_y / 2.0
        val = -3.0 * np.exp(-((y - center)**2) / 25.0)
        return val * (-2 * (y - center) / 25.0)
    if mode_id == 3:
        center = max_y / 2.0
        val = 3.0 * np.exp(-((y - center)**2) / 9.0)
        return val * (-2 * (y - center) / 9.0)
    return 0.0

@njit(parallel=HAVE_NUMBA, fastmath=False)
def simulate_rays_kernel(
    num_pings, 
    ray_angles, 
    ray_gains_db, 
    tl_table, 
    step_size, 
    dr, max_range, 
    water_depth, 
    bathy_mode_id, 
    targets_arr, 
    texture_linear, 
    texture_raw, 
    sb_mu, sb_n, 
    sl, 
    k_wavenumber
):
    max_bins = int(max_range / dr)
    sonar_image = np.zeros((num_pings, max_bins), dtype=np.float64)

    num_rays = len(ray_angles)
    ray_dys = np.cos(ray_angles)
    ray_dzs = np.sin(ray_angles)

    # Gaussian Splatting kernel for sub-bin energy accumulation
    sigma_bins = 0.6 
    inv_2sigma2 = 1.0 / (2 * sigma_bins**2)
    max_tl_idx = len(tl_table) - 1

    # Normalization factor for ray density to keep energy physical
    ray_norm = 1.0 / num_rays

    for p in prange(num_pings):
        ship_x = p * step_size

        for i in range(num_rays):
            ray_gain_db = ray_gains_db[i]
            dy = ray_dys[i]
            dz = ray_dzs[i]
            if dz <= 1e-9: continue

            # --- 1. ANALYTIC TARGET PRE-CHECK ---
            t_target_min = 1e9
            hit_target_idx = -1

            num_targets = targets_arr.shape[0]
            for t_idx in range(num_targets):
                t_type = int(targets_arr[t_idx, 0])
                t_x = targets_arr[t_idx, 1]
                t_y = targets_arr[t_idx, 2]
                t_size = targets_arr[t_idx, 3]
                t_lift = targets_arr[t_idx, 4]

                # Assume target is near the seabed for coarse check
                bathy_at_target = get_bathymetry_offset_jit(t_y, bathy_mode_id, 50.0)
                z_floor_at_target = water_depth + bathy_at_target
                z_bot = z_floor_at_target - t_lift
                z_top = z_bot - t_size
                z_center = z_bot - (t_size / 2.0)

                t_enter = 1e9
                if t_type == 0: # Sphere
                    radius = t_size / 2.0
                    oc_x = ship_x - t_x
                    oc_y = 0.0 - t_y
                    oc_z = 0.0 - z_center
                    b = 2.0 * (dy * oc_y + dz * oc_z)
                    c = (oc_x**2 + oc_y**2 + oc_z**2) - radius**2
                    delta = b*b - 4*c
                    if delta >= 0:
                        t_cand = (-b - np.sqrt(delta)) / 2.0
                        if t_cand > 0: t_enter = t_cand    
                else: # Box
                    # Slab intersection method
                    if abs(ship_x - t_x) <= (t_size / 2.0):
                        t1 = (t_y - t_size/2.0) / dy; t2 = (t_y + t_size/2.0) / dy
                        t3 = (z_top) / dz; t4 = (z_bot) / dz
                        t_enter_slab = max(min(t1, t2), min(t3, t4))
                        t_exit_slab = min(max(t1, t2), max(t3, t4))
                        if t_exit_slab >= t_enter_slab and t_enter_slab > 0:
                            t_enter = t_enter_slab

                if t_enter < t_target_min:
                    t_target_min = t_enter
                    hit_target_idx = t_idx

            # --- 2. RAY MARCHING ---
            current_dist = 0.0
            max_march_steps = int(max_range / dr)

            # OPTIMIZATION: Jump to near target or floor to save cycles
            dist_to_floor = water_depth / dz
            start_dist = min(dist_to_floor, t_target_min) - 5.0
            if start_dist < 0: start_dist = 0.0

            # Align start_dist to dr grid
            step_start = int(start_dist / dr)
            current_dist = step_start * dr

            for step in range(step_start, max_march_steps):
                current_dist += dr
                py = current_dist * dy
                pz = current_dist * dz

                # A. Target Hit
                if current_dist >= t_target_min:
                    t_idx = hit_target_idx
                    t_type = int(targets_arr[t_idx, 0])
                    t_x = targets_arr[t_idx, 1]
                    t_y = targets_arr[t_idx, 2]
                    t_size = targets_arr[t_idx, 3]
                    t_lift = targets_arr[t_idx, 4]

                    bathy_at_target = get_bathymetry_offset_jit(t_y, bathy_mode_id, 50.0)
                    z_floor_at_target = water_depth + bathy_at_target
                    z_bot = z_floor_at_target - t_lift
                    z_center = z_bot - t_size/2.0

                    ny, nz = 0.0, -1.0
                    if t_type == 0:
                        ny = py - t_y; nz = pz - z_center
                        n_len = np.sqrt(ny*ny + nz*nz)
                        if n_len > 0: ny/=n_len; nz/=n_len
                    else:
                        if abs(py - (t_y - t_size/2)) < dr: ny, nz = -1.0, 0.0
                        elif abs(py - (t_y + t_size/2)) < dr: ny, nz = 1.0, 0.0
                        else: ny, nz = 0.0, -1.0

                    cos_gamma = max(0.0, min(1.0, (-dy * ny) + (-dz * nz)))
                    gamma_deg = np.degrees(np.arccos(cos_gamma)) if cos_gamma > 0 else 90.0

                    ts_val = 0.0
                    if t_type == 0:
                        a = t_size / 2.0; ka = k_wavenumber * a
                        sigma = np.pi * a**2 * (ka**4) if ka < 1.0 else np.pi * a**2
                        ts_val = 10 * np.log10(sigma / (4 * np.pi) + 1e-12)
                    else:
                        lam = 2 * np.pi / k_wavenumber; area = t_size * t_size
                        sigma_peak = 4 * np.pi * (area**2) / (lam**2 + 1e-9)
                        beta = (k_wavenumber * t_size * np.sin(np.radians(gamma_deg))) / 2.0
                        sinc_sq = 1.0 if abs(beta) < 1e-6 else (np.sin(beta) / beta)**2
                        sigma = sigma_peak * sinc_sq * (cos_gamma**2)
                        ts_val = 10 * np.log10(sigma / (4 * np.pi) + 1e-12)

                    BS = max(10*np.log10(10**(ts_val/10) + 1e-9), -100.0)
                    dist_idx = min(int((current_dist + 1e-5) / dr) - 1, max_tl_idx)
                    TL = tl_table[dist_idx if dist_idx >=0 else 0]
                    RL = sl - TL + BS + ray_gain_db
                    energy = 10**(RL/10.0)

                    exact_bin = current_dist / dr
                    center_int = int(exact_bin + 0.5)

                    for off in [-1, 0, 1]:
                        idx = center_int + off
                        if idx >= 0 and idx < max_bins:
                            w = np.exp(-(exact_bin - idx)**2 * inv_2sigma2)
                            sonar_image[p, idx] += energy * w * ray_norm
                    break

                # B. Floor Hit
                tex_idx = (i * 127 + step * 31) % max_bins
                local_bathy_off = get_bathymetry_offset_jit(py, bathy_mode_id, 50.0)
                local_depth = water_depth + local_bathy_off + (texture_raw[p, tex_idx] * 0.1)

                if pz >= local_depth:
                    local_slope = get_bathymetry_slope_jit(py, bathy_mode_id, 50.0)
                    local_slope += texture_raw[p, tex_idx] * 0.1
                    norm_n = np.sqrt(local_slope**2 + 1)
                    n_y = -local_slope / norm_n; n_z = 1.0 / norm_n
                    cos_inc = max(-1.0, min(1.0, (dy * n_y) + (dz * n_z)))
                    grazing_rad = np.pi/2 - np.arccos(cos_inc)

                    if grazing_rad > 0:
                        # --- PHYSICS UPGRADE: Continuous Specular + Lambertian ---
                        BS_lambert = sb_mu + sb_n * np.log10(np.sin(grazing_rad) + 1e-6)

                        inc_angle_rad = np.pi/2 - grazing_rad
                        specular_boost = 15.0 * np.exp(-(inc_angle_rad**2) / 0.03)

                        BS = BS_lambert + specular_boost

                        dist_idx = min(int((current_dist + 1e-5) / dr) - 1, max_tl_idx)
                        TL = tl_table[dist_idx if dist_idx >= 0 else 0]
                        RL = sl - TL + BS + ray_gain_db
                        energy = 10**(RL/10.0)

                        bin_idx_tex = int(current_dist / dr)
                        if bin_idx_tex < max_bins: energy *= texture_linear[p, bin_idx_tex]

                        exact_bin = current_dist / dr
                        center_int = int(exact_bin + 0.5)
                        for off in [-1, 0, 1]:
                            idx = center_int + off
                            if idx >= 0 and idx < max_bins:
                                w = np.exp(-(exact_bin - idx)**2 * inv_2sigma2)
                                sonar_image[p, idx] += energy * w * ray_norm
                    break

    return sonar_image

# ==============================================================================
# PART 3: PHYSICS CLASSES & ORCHESTRATION (PRESERVED)
# ==============================================================================

class OceanMedium:
    def __init__(self, temperature_c=15, salinity_ppt=35, ph=8.0, depth_m=50):
        self.T = temperature_c; self.S = salinity_ppt; self.pH = ph; self.D = depth_m
        self.c = 1449.2 + 4.6*self.T - 0.055*self.T**2 + 0.00029*self.T**3 + \
                 (1.34 - 0.01*self.T)*(self.S - 35) + 0.016*self.D
    def get_absorption_db_per_km(self, freq_khz):
        f1 = 0.78 * np.sqrt(self.S/35) * np.exp(self.T/26); A1 = 0.003 * (self.pH - 8)
        term1 = (A1 * f1 * freq_khz**2) / (f1**2 + freq_khz**2)
        f2 = 42 * np.exp(self.T/17); A2 = 0.000175 * (self.S/35) * (1 + 0.025*self.T)
        term2 = (A2 * f2 * freq_khz**2) / (f2**2 + freq_khz**2)
        if self.T <= 20: A3 = 4.937e-4 - 2.59e-5 * self.T + 9.11e-7 * self.T**2 - 1.5e-8 * self.T**3
        else: A3 = 3.964e-4 * np.exp(-5.35e-2 * (self.T - 20))
        term3 = A3 * freq_khz**2
        return term1 + term2 + term3

class SonarSystem:
    def __init__(self, freq_khz, beamwidth_vert_deg, aperture_horiz_m, cycles, tilt_angle_deg, max_range_override=None, medium=None):
        self.freq_khz = freq_khz
        self.freq_hz = freq_khz * 1000.0
        c = medium.c if medium else 1500.0
        self.wavelength = c / self.freq_hz
        self.k = 2 * np.pi / self.wavelength
        self.pulse_len_s = cycles / self.freq_hz
        self.dr = c * self.pulse_len_s / 2.0 
        self.aperture_m = aperture_horiz_m
        self.bw_horiz_rad = self.wavelength / self.aperture_m
        self.bw_horiz_deg = np.degrees(self.bw_horiz_rad)
        self.bw_vert_deg = beamwidth_vert_deg
        self.bw_vert_rad = np.radians(beamwidth_vert_deg)
        self.vb_sigma = self.bw_vert_rad / 2.3548200450309493
        self.tilt_rad = np.radians(tilt_angle_deg)
        self.sl = 210.0; self.noise_floor = 50.0 
        auto_range = 40000.0 / self.freq_khz
        if auto_range < 20: auto_range = 20
        self.max_range = max_range_override if max_range_override else auto_range

    def get_beam_sensitivity(self, angle_from_axis):
        return np.exp(- (angle_from_axis**2) / (2 * self.vb_sigma**2))

class Target:
    def __init__(self, type, x, y, size_m, lift_m=0.0):
        self.type = type; self.x = x; self.y = y; self.size = size_m; self.lift = lift_m

def run_physics_simulation(sonar, medium, targets, water_depth, ship_speed_mps, ping_rate_hz, 
                    bottom_type="Sand", bottom_profile="Flat",
                    bottom_roughness=1.0, enable_targets=True, track_len=20.0):

    seabed_db = {
        "Mud":   {'mu': -35.0, 'n': 10.0},
        "Sand":  {'mu': -28.0, 'n': 15.0},
        "Gravel":{'mu': -20.0, 'n': 30.0},
        "Rock":  {'mu': -18.0, 'n': 25.0}
    }
    sb_params = seabed_db.get(bottom_type, seabed_db["Sand"])

    # Unit is now m/s coming in, so no conversion needed
    step_size = ship_speed_mps / ping_rate_hz
    num_pings = int(track_len / step_size)
    if num_pings < 1: num_pings = 1
    max_bins = int(sonar.max_range / sonar.dr)

    alpha = medium.get_absorption_db_per_km(sonar.freq_khz)

    half_bw = (sonar.bw_vert_rad / 2.0) * 1.1 
    start_angle = sonar.tilt_rad - half_bw
    end_angle = sonar.tilt_rad + half_bw
    if start_angle < 0.01: start_angle = 0.01

    rays_per_beam = 150 # Increased density for smoother fills
    delta_theta_rad = sonar.bw_vert_rad / rays_per_beam
    num_rays = int((end_angle - start_angle) / delta_theta_rad)
    if num_rays < 50: num_rays = 50
    ray_angles = np.linspace(start_angle, end_angle, num_rays)

    ray_gains = []
    for angle in ray_angles:
        off_axis = angle - sonar.tilt_rad
        if abs(off_axis) > half_bw: ray_gains.append(0.0)
        else: ray_gains.append(sonar.get_beam_sensitivity(off_axis))
    ray_gains = np.array(ray_gains, dtype=np.float64)
    ray_gains_db = 10 * np.log10(ray_gains + 1e-9)

    # TL Calculation
    max_steps = int(sonar.max_range / sonar.dr) + 2
    steps_arr = np.arange(1, max_steps + 1, dtype=np.float64)
    dists_arr = steps_arr * sonar.dr
    tl_table = 40 * np.log10(dists_arr) + 2 * (alpha / 1000.0) * dists_arr

    texture_map = np.random.normal(0, bottom_roughness, (num_pings, max_bins))
    texture_linear = 10**(texture_map / 10.0)

    target_data = []
    if enable_targets:
        for t in targets:
            t_id = 1 if t.type == "Box" else 0
            target_data.append([t_id, t.x, t.y, t.size, t.lift])
    targets_arr = np.array(target_data, dtype=np.float64)
    if len(target_data) == 0: targets_arr = np.zeros((0, 5), dtype=np.float64)

    b_map = {"Flat": 0, "Slope": 1, "Mound": 2, "Trench": 3}
    b_mode_id = b_map.get(bottom_profile, 0)

    # --- KERNEL EXECUTION (PURE PHYSICS) ---
    sonar_image = simulate_rays_kernel(
        num_pings, ray_angles, ray_gains_db, tl_table, 
        step_size, sonar.dr, sonar.max_range,
        water_depth, b_mode_id, targets_arr, texture_linear, texture_map,
        float(sb_params['mu']), float(sb_params['n']), float(sonar.sl), float(sonar.k)
    )

    # --- RECEIVER CHAIN (SIGNAL CONDITIONING) ---
    # 1. Near-Field Blanking (Receiver Ring-Down)
    ring_down_bins = int(2.0 / sonar.dr)
    if ring_down_bins > 0 and ring_down_bins < max_bins:
        sonar_image[:, :ring_down_bins] *= 0.05

    # 2. Additive Noise (Before Normalization)
    noise_pwr = 10**(sonar.noise_floor/10.0)
    sonar_image += noise_pwr

    # 3. Dynamic Range Normalization (Fixed Reference)
    min_ref_level = noise_pwr * 1e6
    signal_ref_level = np.percentile(sonar_image, 99.0)
    reference_level = max(signal_ref_level, min_ref_level)

    sonar_image /= reference_level

    final_image = sonar_image 

    if sonar.bw_horiz_deg > 0.001:
        range_axis = np.arange(max_bins) * sonar.dr
        footprint_m = range_axis * sonar.bw_horiz_rad
        footprint_pings = footprint_m / step_size
        final_image = np.zeros_like(sonar_image)
        for b in range(max_bins):
            w = int(footprint_pings[b])
            if w < 1: 
                final_image[:, b] = sonar_image[:, b]
            else:
                col = sonar_image[:, b]
                kernel = np.ones(w) / w 
                conv_res = np.convolve(col, kernel, mode='full')[:len(col)]
                final_image[:, b] = conv_res

    amp_image = np.sqrt(final_image)
    noisy_amp = np.random.rayleigh(scale=amp_image)

    raw_power_output = noisy_amp ** 2
    return raw_power_output, sonar.dr, step_size, num_pings, alpha

# ==============================================================================
# PART 4: DISPLAY & POST-PROCESSING LAYER (STRICTLY STATELESS)
# ==============================================================================

def apply_post_processing(raw_img_input, display_params, physics_metadata):
    """
    Apply pure visualization transforms (Log, Contrast, Brightness).
    NO adaptive logic allowed.
    """
    # 1. Work on a copy
    working_img = raw_img_input.copy()

    # 2. Log Compression
    img_db = 10 * np.log10(working_img + 1e-10)

    # 3. Brightness & Contrast (Linear Viz Only)
    contrast = display_params.get('contrast', 1.0)
    brightness = display_params.get('brightness', 0.0)
    img_db = (img_db * contrast) + brightness

    # 4. Final Clipping
    vmin = display_params.get('clip_min', -60.0)
    vmax = display_params.get('clip_max', 0.0)
    img_final = np.clip(img_db, vmin, vmax)

    return img_final

# ==============================================================================
# PART 5: STREAMLIT UI (INVALIDATION LOGIC)
# ==============================================================================

if 'sim_result' not in st.session_state:
    st.session_state.sim_result = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

# Callback to force re-simulation if physics params change
def clear_sim():
    st.session_state.raw_data = None
    st.session_state.sim_result = None

st.sidebar.title("SonarSim Pro")

# PROGRESS BAR PLACEHOLDER
progress_bar = st.sidebar.progress(0)

st.sidebar.markdown("---")

# --- PHYSICS CONTROLS ---
with st.sidebar.expander("1. Physics Engine (Sim)", expanded=True):
    # All physics widgets trigger clear_sim on change
    # UPDATED: Speed in m/s
    speed_mps = st.slider("Speed (m/s)", 0.5, 5.0, 1.5, on_change=clear_sim)
    ping_rate = st.slider("Ping Rate (Hz)", 5.0, 30.0, 15.0, on_change=clear_sim)
    aperture_len = st.number_input("Horiz Aperture (m)", 0.05, 1.0, 0.215, on_change=clear_sim)
    freq = st.number_input("Freq (kHz)", 50.0, 1000.0, 400.0, on_change=clear_sim)

    c1, c2 = st.columns(2)
    with c1: b_type = st.selectbox("Seabed", ["Mud", "Sand", "Gravel", "Rock"], index=1, on_change=clear_sim)
    with c2: b_profile = st.selectbox("Bathymetry", ["Flat", "Slope", "Mound", "Trench"], on_change=clear_sim)
    roughness = st.slider("Texture (dB)", 0.0, 5.0, 1.0, on_change=clear_sim)

    limit_r = 40000.0 / freq
    default_r = float(min(limit_r, 600.0))
    max_r = st.number_input("Display Range (m)", 10.0, 1000.0, default_r, on_change=clear_sim)

    water_depth = st.number_input("Water Depth (m)", 5.0, 100.0, 15.0, on_change=clear_sim)
    tilt_angle = st.slider("Tilt Angle (deg)", 0, 80, 46, on_change=clear_sim)
    beam_v = st.slider("Vert Beam (deg)", 10, 90, 36, on_change=clear_sim)
    num_cycles = st.slider("Cycles (N)", 2, 100, 40, on_change=clear_sim)

    st.markdown("**Targets**")
    show_targets = st.checkbox("Enable Targets", value=True, on_change=clear_sim)
    t1_type = st.selectbox("Type", ["Sphere", "Box"], on_change=clear_sim)

    # KEYED SLIDER to preserve state
    # Ensure initialization once
    if "target_y_pos" not in st.session_state:
        st.session_state.target_y_pos = 15.0

    # Clamp BEFORE widget creation
    st.session_state.target_y_pos = min(
        st.session_state.target_y_pos,
        float(max_r)
    )

    t1_y = st.slider(
        "Cross Track Y",
        0.0,
        float(max_r),
        st.session_state.target_y_pos,
        key="target_y_pos",
        on_change=clear_sim
    )

    t1_s = st.number_input("Size (m)", 0.1, 5.0, 1.0, on_change=clear_sim)
    t1_l = st.slider("Target Lift (m)", 0.0, 10.0, 0.0, on_change=clear_sim)

    run_btn = st.button("RUN SIMULATION", type="primary")

# --- DISPLAY CONTROLS (PURE VIZ) ---
with st.sidebar.expander("2. Visualization", expanded=True):
    # These controls do NOT clear simulation
    contrast = st.slider("Contrast", 0.5, 3.0, 1.0)
    brightness = st.slider("Brightness (dB Offset)", -50.0, 50.0, 0.0)
    c_min, c_max = st.slider("dB Clipping Range", -100.0, 20.0, (-60.0, 0.0))

display_params = {
    'contrast': contrast,
    'brightness': brightness,
    'clip_min': c_min,
    'clip_max': c_max
}

# --- DERIVED PHYSICS PARAMS ---
c_approx = 1500.0
pulse_len_us = (num_cycles / (freq * 1000.0)) * 1e6
wavelength_approx = c_approx / (freq * 1000.0)
beam_h_deg_disp = np.degrees(wavelength_approx / aperture_len)
tilt_rad = np.radians(tilt_angle)
half_beam_rad = np.radians(beam_v / 2)

if run_btn:
    progress_bar.progress(10)
    med = OceanMedium(depth_m=water_depth)
    son = SonarSystem(freq_khz=freq, 
                      beamwidth_vert_deg=beam_v, 
                      aperture_horiz_m=aperture_len, 
                      cycles=num_cycles, 
                      tilt_angle_deg=tilt_angle, 
                      max_range_override=max_r, 
                      medium=med)

    tgt = Target(t1_type, 10.0, t1_y, t1_s, lift_m=t1_l)

    progress_bar.progress(30)
    with st.spinner("Simulating Physics..."):
        # Pass MPS directly
        raw, dr, dx, n_pings, alpha_val = run_physics_simulation(
            son, med, [tgt], water_depth, 
            ship_speed_mps=speed_mps, 
            ping_rate_hz=ping_rate,
            bottom_type=b_type,
            bottom_profile=b_profile,
            bottom_roughness=roughness,
            enable_targets=show_targets,
            track_len=20.0
        )

    progress_bar.progress(100)
    st.session_state.raw_data = raw
    st.session_state.sim_result = {'dr': dr, 'dx': dx, 'np': n_pings, 'alpha': alpha_val}
    time.sleep(0.5) # Small visual delay for completion
    progress_bar.empty()
    st.rerun()

# --- RENDERING ---
sim_track_len = 20.0
top_ray_angle = tilt_rad - half_beam_rad 
if top_ray_angle > 0.02: geom_slant_max = water_depth / np.sin(top_ray_angle)
else: geom_slant_max = 40000.0 / freq 
smart_plot_limit = min(geom_slant_max, max_r) * 1.1
if smart_plot_limit > water_depth: plot_y_max = np.sqrt(smart_plot_limit**2 - water_depth**2)
else: plot_y_max = 10.0 
view_depth_limit = water_depth + 5.0 

# REDUCED HSPACE to close gaps
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, width_ratios=[plot_y_max, smart_plot_limit], height_ratios=[view_depth_limit, sim_track_len], wspace=0.1, hspace=0.05)

# 1. CROSS SECTION
ax_cross = fig.add_subplot(gs[0, 0])
# Moved Title to TOP (cleaner layout)
ax_cross.set_title("Cross Section (Y-Z)", fontweight='bold', pad=10)
ax_cross.set_ylabel("Depth (m)")
ax_cross.set_aspect('equal', adjustable='box')
ax_cross.invert_yaxis()

bathy_y = np.linspace(0, plot_y_max, 100)
def visual_bathy_local(y_arr):
    z_out = np.zeros_like(y_arr)
    if b_profile == "Flat": return z_out
    if b_profile == "Slope": return (y_arr / 50.0) * 5.0
    if b_profile == "Mound": return -3.0 * np.exp(-((y_arr - 25.0)**2) / 25.0)
    if b_profile == "Trench": return 3.0 * np.exp(-((y_arr - 25.0)**2) / 9.0)
    return z_out
bathy_z = water_depth + visual_bathy_local(bathy_y)
ax_cross.plot(bathy_y, bathy_z, 'brown', lw=2)
ax_cross.axhline(0, color='blue', ls='--')

# Beam Visualization (Poly + Arcs)
theta1 = tilt_rad - half_beam_rad
theta2 = tilt_rad + half_beam_rad
y1 = smart_plot_limit * np.cos(theta1); z1 = smart_plot_limit * np.sin(theta1)
y2 = smart_plot_limit * np.cos(theta2); z2 = smart_plot_limit * np.sin(theta2)

# Filled Wedge
poly_points = [(0,0), (y1, z1), (y2, z2)]
beam_wedge = patches.Polygon(poly_points, closed=True, color='green', alpha=0.1)
ax_cross.add_patch(beam_wedge)
ax_cross.plot([0, y1], [0, z1], 'g--', alpha=0.3)
ax_cross.plot([0, y2], [0, z2], 'g--', alpha=0.3)

# ADDED: Center Line
center_y = smart_plot_limit * np.cos(tilt_rad)
center_z = smart_plot_limit * np.sin(tilt_rad)
ax_cross.plot([0, center_y], [0, center_z], 'r-.', alpha=0.4, lw=1)

# Clipped Arcs (Range Rings)
arc_theta = np.linspace(theta1, theta2, 20)
for r_circ in range(5, int(smart_plot_limit)+5, 5):
    arc_y = r_circ * np.cos(arc_theta)
    arc_z = r_circ * np.sin(arc_theta)
    ax_cross.plot(arc_y, arc_z, color='gray', linestyle=':', alpha=0.5, lw=0.8)
    if r_circ % 10 == 0 and r_circ < smart_plot_limit:
        # CHANGED: Label on TOP ARM (theta1 is the top-most visual edge)
        lbl_y = r_circ * np.cos(theta1)
        lbl_z = r_circ * np.sin(theta1)
        ax_cross.text(lbl_y, lbl_z, f"{r_circ}", 
                      color='gray', fontsize=8, ha='right', va='bottom')

if show_targets:
    if t1_type == "Box":
        rect = patches.Rectangle((t1_y-t1_s/2, water_depth-t1_l-t1_s), t1_s, t1_s, color='black')
        ax_cross.add_patch(rect)
    else:
        circ = patches.Circle((t1_y, water_depth-t1_l-t1_s/2), t1_s/2, color='black')
        ax_cross.add_patch(circ)

ax_cross.set_xlim(0, plot_y_max)
ax_cross.set_ylim(view_depth_limit, -1) 

# 2. SIMULATION SUMMARY PANEL (Top Right)
ax_info = fig.add_subplot(gs[0, 1])
ax_info.axis('off')
ax_info.set_title("Simulation Summary", fontweight='bold')

if st.session_state.raw_data is not None:
    meta = st.session_state.sim_result
    res_r = meta['dr'] * 100
    res_a = meta['dx'] * 100
    alpha_db = meta['alpha']

    info_text = (
        f"--- ACOUSTICS ---\n"
        f"Frequency:      {freq:.1f} kHz\n"
        f"Wavelength:     {wavelength_approx*100:.2f} cm\n"
        f"Absorbtion:     {alpha_db:.2f} dB/km\n"
        f"Beam (H x V):   {beam_h_deg_disp:.2f}° x {beam_v}°\n"
        f"Pulse:          {num_cycles} cyc ({pulse_len_us:.1f} µs)\n\n"
        f"--- RESOLUTION ---\n"
        f"Range Res:      {res_r:.1f} cm\n"
        f"Along-Track:    {res_a:.1f} cm\n\n"
        f"--- GEOMETRY ---\n"
        f"Max Phy Range:  {limit_r:.0f} m\n"
        f"Disp Range:     {max_r:.1f} m\n"
        f"Depression:     {tilt_angle}°\n"
        f"Ping Rate:      {ping_rate} Hz @ {speed_mps} m/s"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                 verticalalignment='top', fontfamily='monospace', fontsize=10)
else:
    ax_info.text(0.5, 0.5, "Settings Changed.\nPress RUN to simulate.", ha='center', transform=ax_info.transAxes)

# 3. TOP DOWN VIEW (Bottom Left)
ax_top = fig.add_subplot(gs[1, 0], sharex=ax_cross)
# MOVED TITLE DOWN to avoid X-axis conflict
ax_top.text(0.5, -0.25, "Top Down (Ground Range)", transform=ax_top.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
ax_top.set_xlabel("Across Track Y (m)")
ax_top.set_ylabel("Along Track X (m)")
ax_top.set_aspect('equal', adjustable='box')
ax_top.set_xlim(0, plot_y_max)
ax_top.set_ylim(0, sim_track_len)

if show_targets:
    tgt_x_pos = 10.0
    if t1_type == "Box":
        rect2 = patches.Rectangle((t1_y-t1_s/2, tgt_x_pos-t1_s/2), t1_s, t1_s, color='black')
        ax_top.add_patch(rect2)
    else:
        circ2 = patches.Circle((t1_y, tgt_x_pos), t1_s/2, color='black')
        ax_top.add_patch(circ2)

# 4. SONAR WATERFALL (Bottom Right)
ax_sonar = fig.add_subplot(gs[1, 1], sharey=ax_top)
# MOVED TITLE DOWN to avoid X-axis conflict
ax_sonar.text(0.5, -0.25, "Sonar Image (Slant Range)", transform=ax_sonar.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
ax_sonar.set_xlabel(f"Slant Range (m)")
ax_sonar.set_aspect('equal', adjustable='box')
ax_sonar.set_ylim(0, sim_track_len)
ax_sonar.set_xlim(0, smart_plot_limit)

if st.session_state.raw_data is not None:
    # PASS A COPY TO PRESERVE PHYSICS
    img_final = apply_post_processing(
        st.session_state.raw_data.copy(), 
        display_params, 
        st.session_state.sim_result
    )

    ax_sonar.imshow(img_final, origin='lower', 
                    extent=[0, max_r, 0, sim_track_len], 
                    cmap='copper', 
                    vmin=c_min, vmax=c_max)
else:
    # BLACK SCREEN ON INVALIDATION
    ax_sonar.set_facecolor('black')
    ax_sonar.text(0.5, 0.5, "INVALIDATED", color='white', ha='center')

plt.setp(ax_sonar.get_yticklabels(), visible=False)
st.pyplot(fig)
