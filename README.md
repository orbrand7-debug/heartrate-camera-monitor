# C++23 Heartrate HUD Monitor

A high-performance heart rate monitor that analyzes skin hue changes via webcam and displays a transparent HUD overlay compatible with fullscreen DirectX 11/12 games.

## Features
- **C++23 Standard**: Utilizes modern features like `std::expected`, `std::format`, and `std::jthread`.
- **Dlib Landmarks**: Precise forehead ROI extraction using 68-point face landmarks.
- **FFT Analysis**: Hue-based heart rate estimation using Discrete Fourier Transforms.
- **Win32 Overlay**: A transparent, click-through HUD that stays on top of games.
- **Global Hotkeys**: Configurable hotkey (default `Ctrl+Alt+D`) to toggle debug mode.
- **YAML Config**: Fully adjustable via `config.yaml` (Colors, Fonts, BPM range, HUD position).

## Prerequisites
- **Windows 10/11** (Tested on IoT LTSC 19044).
- **Visual Studio 2026** (v144 toolset or newer).
- **vcpkg**: For dependency management.
- **Webcam**: Standard USB or integrated camera.

