# DSP Studio 🎛️
**An Interactive Web-Based Digital Signal Processing Workstation**

DSP Studio is an educational and interactive web application designed to visualize classical Digital Signal Processing (DSP) algorithms end-to-end. Users can upload or record audio, process it through various DSP modules, and see the real-time mathematical transformations via animated, math-labeled visualizations.

This project was developed for the **CSE 220 Digital Signal Processing** course.

**Repository:** [GitHub Link](https://github.com/MdMostakim103/Digital%20Signal%20Processing_Project)

---

## 👥 Authors
*   **Md Mostakim** | ID: 2305063
*   **Samiul Basir Adry** | ID: 2305071

---

## 🚀 Overview: What We Built
The application consists of **8 core processing modules** and a centralized **Studio** to chain effects together.

### 1. Amplitude & Gain
*   **Algorithm:** $y[n] = A \cdot x[n]$
*   **Features:** Scales the input signal by a constant gain factor. Reports Peak and RMS values, and instantly visualizes clipping when gain exceeds full scale.

### 2. Time-Domain Processing
*   **Delay:** Adds a single, discrete repetition without feedback. ($y[n] = x[n] + \alpha \cdot x[n-D]$)
*   **Echo:** Implements a feedback delay line to create several decaying, evenly-spaced echoes.
*   **Convolution:** Computes the mathematical convolution ($y[n] = \sum_k x[k]h[n-k]$) by reversing the impulse response $h[n]$ and sliding it across the input signal.

### 3. Spectral Detection (Pitch)
*   **Algorithm:** Applies a Hann window and computes the FFT to extract the spectrum $|X(f)|$.
*   **Features:** Restricts the search range to ignore DC/rumble, finds the dominant frequency $f_0$, maps it to the nearest musical note, and synthesizes a pure sine wave for auditory comparison.

### 4. Bird Sound Detector (Bonus)
*   **Algorithm:** k-Nearest Neighbors (k-NN) with no ML.
*   **Features:** Analyzes 2-second windows using 6 hand-crafted features (RMS energy, zero-crossing rate, dominant frequency, spectral centroid, bandwidth, rolloff). Achieves 79.4% accuracy across Crow, Robin, and Owl datasets using z-score-normalized distance.

### 5. Sampling & Quantization
*   **Quantization:** Adjusts bit depth ($b$) and computes step sizes ($\Delta$) to round and clip audio samples to specific levels.
*   **Sample-and-Hold:** Simulates a lower sample rate by holding a block's first sample for $N$ duration (zero-order hold).

### 6. Frequency & Filtering
*   **Algorithm:** Designs filters as Second-Order Sections (SOS) to maintain numerical stability.
*   **Features:** Applies filters using `sosfiltfilt` (forwards and backwards) to cancel out phase delay. Supports multiple families (Butterworth, Chebyshev I/II, Elliptic, Bessel, Ideal) and band types (Lowpass, Highpass, Bandpass, Bandstop).

### 7. Voice Morphing (Phase Vocoder)
*   **Algorithm:** Uses Short-Time Fourier Transform (STFT) to split audio into overlapping frames, manipulating magnitude and phase independently.
*   **Features:** Supports pitch-shifting and time-stretching. Includes special presets (Robot, Whisper, Vader, Droid, etc.) that force phase adjustments or layer pitch-shift, filtering, ring-modulation, and reverb.

### 8. Spectral Portal (Spectrogram Masking)
*   **Algorithm:** $X'(k,m) = M(k,m) \cdot X(k,m)$
*   **Features:** Generates a windowed STFT spectrogram. Users can draw "KEEP/ERASE" regions to create a binary mask, which is Gaussian-blurred (feathered) to avoid audio ringing, and then reconstructed via Inverse STFT.

### 🎛️ The Studio (Multi-Operation Signal Chain)
*   Allows users to upload or record one input signal and build a custom effects chain (up to 25 steps).
*   Any of the 7 modules' effects can be added, reordered, or removed.
*   The output of one step seamlessly feeds into the input of the next.

---

## 🛠️ Tech Stack
**Backend Engine**
*   Python
*   FastAPI
*   NumPy
*   SciPy
*   Librosa
*   soundfile

**Frontend Interaction**
*   React
*   Vite
*   Modern CSS3
*   Canvas-based live visualizations

---

## 📚 DSP Methods Covered
*   FFT & Spectrograms
*   Convolution
*   IIR/FIR Filtering
*   Quantization & Aliasing
*   Phase Vocoder
*   Spectral Masking