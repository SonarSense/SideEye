# SideEye: Starboard Side Scan Sonar Simulator

**SideEye** is a physics-based acoustic simulation engine built with Python and Streamlit. Unlike simple noise-based visualizers, SideEye uses ray tracing and the active sonar equation to simulate how sound interacts with the seabed, targets, and water column to generate realistic Side Scan Sonar (SSS) waterfall images.

This tool is designed for enthusiasts who want to understand the relationship between sensor parameters (frequency, beamwidth, pulse length) and image quality.

## Features

### 1. Physics-Based Rendering
* **Ray Tracing Kernel:** Simulates acoustic ray paths accounting for grazing angles, beam patterns, and travel time.
* **The Sonar Equation:** Accurately models Source Level, Transmission Loss (geometric spreading + frequency-dependent absorption), Target Strength, and Noise.
* **Acoustic Shadows:** Calculates realistic shadows based on target lift, size, and the sensor's altitude above the seafloor.
* **Seabed Properties:** Adjust scattering strength for different seabed types (Mud, Sand, Gravel, Rock) using Lambertian scattering models.

### 2. Real-Time Parameter Tuning
* **Sensor Configuration:** Modify Frequency (kHz), Horizontal/Vertical Beamwidths, Pulse Length (Cycles), and Tilt Angle.
* **Platform Dynamics:** Adjust Tow Speed and Ping Rate to visualize the effects of spatial aliasing or gaps in coverage.
* **Environment:** Change water depth and bathymetry profiles (Flat, Slope, Mound, Trench).

### 3. Performance
* **JIT Compilation:** Utilizes **Numba** to compile critical ray-marching loops, ensuring fast execution even with thousands of rays per ping.

### 4. Visualization Toolkit
* **Cross-Section View (Y-Z):** Visualizes the beam geometry and seabed profile.
* **Top-Down View (X-Y):** Shows the ground truth location of targets.
* **Simulated Waterfall:** The resulting sonar image with adjustable post-processing (Contrast, Brightness, Clipping).

## Installation

### Prerequisites
* Python 3.8 or higher

### 1. Clone the repository

git clone [https://github.com/SonarSense/SideEye.git](https://github.com/SonarSense/SideEye.git)
cd SideEye

2. Install Dependencies
It is recommended to use a virtual environment.

pip install -r requirements.txt


3. Run the Application
streamlit run sonar_app.py

The application should automatically open in your default web browser at http://localhost:8501.

Usage Guide
Physics Engine (Sidebar)
Speed & Ping Rate: These control your Along-Track resolution.

Note: The simulator will warn you if your ping rate is too high (Ghost Echoes) or too low (Gaps in coverage).

Freq (kHz): Higher frequency results in better resolution but higher absorption (shorter range).

Horiz Aperture: Controls the physical length of the array. A larger aperture creates a narrower horizontal beam, improving azimuth resolution.

Targets: You can toggle targets on/off.

Target Lift: Raising a target off the seafloor separates the object from its acoustic shadow.

Visualization (Sidebar)
Contrast/Brightness: Adjust these after running the simulation to interpret the acoustic data, similar to the gain knobs on a real sonar topside unit.

Technical Details
SideEye solves the active sonar equation for every sample bin:

RL = SL - 2TL + TS - NL + DI

Where:

SL: Source Level (Fixed at 210 dB ref 1µPa)

TL: Transmission Loss (Spherical spreading + Frequency dependent absorption derived from temperature/salinity)

TS: Target Strength (Calculated via physical cross-section for targets or Lambertian scattering for the seabed)

NL: Noise Level

DI: Directivity Index (Beam patterns modeled as Gaussian functions)

Dependencies
streamlit: For the web interface.

numpy: For array manipulation and math.

matplotlib: For plotting the geometry and sonar images.

numba: For JIT compilation of the ray-tracing physics kernel.

License
MIT License


### Don't forget your `requirements.txt`

To make the installation instructions in the README work, ensure you have a file named `requirements.txt` in the same folder with the following content:

text
streamlit
numpy
matplotlib
numba
