"""Standalone script to render the SideEye simulation output without Streamlit."""
import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# Extract and execute physics code from sonar_app.py (Parts 1-4 only)
with open('sonar_app.py', 'r') as f:
    source = f.read()
start = source.index("@njit(fastmath=False)\ndef get_bathymetry_offset_jit")
end = source.index("# PART 5: STREAMLIT UI")
exec(compile(source[start:end], 'sonar_app.py', 'exec'))

# --- Default parameters ---
freq, beam_v, aperture_len, num_cycles, tilt_angle = 600.0, 36, 0.215, 40, 46
water_depth, water_temp, water_salinity = 15.0, 15.0, 35.0
speed_mps, ping_rate = 1.5, 10.0
b_type, b_profile, roughness = "Sand", "Mound", 1.0
show_targets, t1_type, t1_y, t1_s, t1_l = True, "Sphere", 10.0, 1.0, 4.0

# --- Derived values ---
med = OceanMedium(temperature_c=water_temp, salinity_ppt=water_salinity, depth_m=water_depth)
c_sound = med.c
wavelength = c_sound / (freq * 1000.0)
beam_h_rad = wavelength / aperture_len
beam_h_deg = np.degrees(beam_h_rad)
dr = c_sound * (num_cycles / (freq * 1000.0)) / 2.0
dx = speed_mps / ping_rate
max_r = float(min(40000.0 / freq, 600.0))

son = SonarSystem(freq_khz=freq, beamwidth_vert_deg=beam_v, aperture_horiz_m=aperture_len,
                  cycles=num_cycles, tilt_angle_deg=tilt_angle, max_range_override=max_r, medium=med)
tgt = Target(t1_type, 10.0, t1_y, t1_s, lift_m=t1_l)

print(f"Sound speed: {c_sound:.1f} m/s | Range res: {dr*100:.1f} cm | Max range: {max_r:.1f} m")
print("Running simulation...")

raw, _, _, n_pings, alpha_val = run_physics_simulation(
    son, med, [tgt], water_depth, ship_speed_mps=speed_mps, ping_rate_hz=ping_rate,
    bottom_type=b_type, bottom_profile=b_profile, bottom_roughness=roughness,
    enable_targets=show_targets, track_len=20.0)
print(f"Done: {raw.shape[0]} pings x {raw.shape[1]} bins")

img_final = apply_post_processing(raw.copy(),
    {'contrast': 1.0, 'brightness': -10.0, 'clip_min': -60.0, 'clip_max': 0.0}, {})

# --- Plot (same layout as app) ---
sim_track_len = 20.0
tilt_rad, half_beam_rad = np.radians(tilt_angle), np.radians(beam_v / 2)
top_ray_angle = tilt_rad - half_beam_rad
geom_slant_max = water_depth / np.sin(top_ray_angle) if top_ray_angle > 0.02 else 40000.0/freq
smart_plot_limit = min(geom_slant_max, max_r) * 1.1
plot_y_max = np.sqrt(smart_plot_limit**2 - water_depth**2) if smart_plot_limit > water_depth else 10.0
view_depth_limit = water_depth + 5.0
beam_width_at_max = max_r * beam_h_rad
limit_ping_rate = c_sound / (2.0 * max_r)
limit_min_rate = speed_mps / beam_width_at_max if beam_width_at_max > 0 else 999.0

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, width_ratios=[plot_y_max, smart_plot_limit],
             height_ratios=[view_depth_limit, sim_track_len], wspace=0.1, hspace=0.05)

# Cross Section
ax_cross = fig.add_subplot(gs[0, 0])
ax_cross.set_title("Cross Section (Y-Z)", fontweight='bold', pad=10)
ax_cross.set_ylabel("Depth Z (m)"); ax_cross.set_aspect('equal', adjustable='box'); ax_cross.invert_yaxis()
bathy_y = np.linspace(0, plot_y_max, 100)
center = plot_y_max / 2.0
bathy_z = water_depth + (-3.0 * np.exp(-((bathy_y - center)**2) / 25.0))
ax_cross.plot(bathy_y, bathy_z, 'brown', lw=2); ax_cross.axhline(0, color='blue', ls='--')
theta1, theta2 = tilt_rad - half_beam_rad, tilt_rad + half_beam_rad
y1, z1 = smart_plot_limit * np.cos(theta1), smart_plot_limit * np.sin(theta1)
y2, z2 = smart_plot_limit * np.cos(theta2), smart_plot_limit * np.sin(theta2)
ax_cross.add_patch(patches.Polygon([(0,0),(y1,z1),(y2,z2)], closed=True, color='green', alpha=0.1))
ax_cross.plot([0,y1],[0,z1], 'g--', alpha=0.3); ax_cross.plot([0,y2],[0,z2], 'g--', alpha=0.3)
cy, cz = smart_plot_limit*np.cos(tilt_rad), smart_plot_limit*np.sin(tilt_rad)
ax_cross.plot([0,cy],[0,cz], 'r-.', alpha=0.4, lw=1)
arc_theta = np.linspace(theta1, theta2, 20)
for r_circ in range(5, int(smart_plot_limit)+5, 5):
    ax_cross.plot(r_circ*np.cos(arc_theta), r_circ*np.sin(arc_theta), 'gray', ls=':', alpha=0.5, lw=0.8)
    if r_circ % 10 == 0 and r_circ < smart_plot_limit:
        ax_cross.text(r_circ*np.cos(theta1), r_circ*np.sin(theta1), f"{r_circ}", color='gray', fontsize=8, ha='right')
ax_cross.add_patch(patches.Circle((t1_y, water_depth-t1_l-t1_s/2), t1_s/2, color='black'))
ax_cross.set_xlim(0, plot_y_max); ax_cross.set_ylim(view_depth_limit, -1)

# Summary
ax_info = fig.add_subplot(gs[0, 1]); ax_info.axis('off'); ax_info.set_title("Simulation Summary", fontweight='bold')
warnings, status_color = [], "green"
if ping_rate > limit_ping_rate: warnings.append(f"[!] TOO FAST\n    Max: {limit_ping_rate:.1f} Hz"); status_color="red"
elif ping_rate < limit_min_rate: warnings.append(f"[!] TOO SLOW\n    Min: {limit_min_rate:.1f} Hz"); status_color="orange"
if not warnings: warnings.append("Parameters Optimal\n    No aliasing or gaps.")
info = f"--- STATUS: {status_color.upper()} ---\n" + "\n".join(warnings)
info += f"\n\n--- CONFIG ---\nFreq: {freq:.0f} kHz | Range: {max_r:.0f} m\nTilt: {tilt_angle} deg | Beam: {beam_h_deg:.2f} x {beam_v} deg\nSound Speed: {c_sound:.1f} m/s"
info += f"\n\n--- RESOLUTION ---\nRange: {dr*100:.1f} cm | Azimuth: {beam_width_at_max:.2f} m\nPing Spacing: {dx*100:.1f} cm"
ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes, va='top', fontfamily='monospace', fontsize=10,
             bbox=dict(facecolor='white' if status_color=='green' else '#ffcccc', alpha=0.5, edgecolor='none'))

# Top Down
ax_top = fig.add_subplot(gs[1, 0], sharex=ax_cross)
ax_top.text(0.5, -0.25, "Top Down (Ground Range)", transform=ax_top.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
ax_top.set_xlabel("Across Track Y (m)"); ax_top.set_ylabel("Along Track X (m)")
ax_top.set_aspect('equal', adjustable='box'); ax_top.set_xlim(0, plot_y_max); ax_top.set_ylim(0, sim_track_len)
ax_top.add_patch(patches.Circle((t1_y, 10.0), t1_s/2, color='black'))

# Sonar Image
ax_sonar = fig.add_subplot(gs[1, 1], sharey=ax_top)
ax_sonar.text(0.5, -0.25, "Sonar Image (Slant Range)", transform=ax_sonar.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
ax_sonar.set_xlabel("Slant Range R (m)"); ax_sonar.set_aspect('equal', adjustable='box')
ax_sonar.set_ylim(0, sim_track_len); ax_sonar.set_xlim(0, smart_plot_limit)
ax_sonar.imshow(img_final, origin='lower', extent=[0, max_r, 0, sim_track_len], cmap='copper', vmin=-60, vmax=0)
plt.setp(ax_sonar.get_yticklabels(), visible=False)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sideeye_output.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0E1117', edgecolor='none')
print(f"Saved to {output_path}")
plt.close()
