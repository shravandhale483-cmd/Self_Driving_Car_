# Autonomous + Voice-Controlled Self-Driving Car

End-to-end pipeline: dataset generation → CNN training → autonomous driving via Socket.IO — with real-time **voice command override** using Whisper STT.

---

## Structure

```
Nirmaan/
├── data/
│   ├── IMG/                # Camera frames
│   └── driving_log.csv     # Telemetry log (steering, throttle, brake, speed)
├── generate_dataset.py     # Synthetic dataset generator
├── train.py                # Train Nvidia CNN → exports model.h5
├── drive.py                # Socket server (port 4567) — autonomous + voice control
├── simulator.py            # Track simulator (manual & autonomous)
├── run_autonomous.py       # Single-command automated orchestration runner
├── model.h5                # Trained model weights
├── requirements.txt
└── README.md
```

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Generate dataset**
```bash
python generate_dataset.py
```
*Or: `python simulator.py` → press `M` (manual mode) + `R` (record)*

**3. Train model**
```bash
python train.py
```
Outputs `model.h5`.

---

## 🤖 Running Autonomously

Once you have a trained `model.h5`, you can run the self-driving car autonomously. You can choose between the **automated single-command run** (recommended) or launching the components **manually in separate terminals**.

### Option A: Automated Run (Single Command - Recommended)
The project includes a runner script, [run_autonomous.py](file:///c:/Users/athar/OneDrive/Desktop/Nirmaan/run_autonomous.py), that automatically starts the Socket.IO drive server (`drive.py`), waits for it to initialize and load the model, and then launches the built-in Pygame simulator (`simulator.py`) in autonomous mode.

To run:
```bash
python run_autonomous.py
```
*   **To Exit:** Close the Pygame simulator window, or press `Ctrl+C` in your terminal window. The runner script will gracefully terminate all background processes for you.

---

### Option B: Manual Run (Separate Terminals)
If you wish to view separate logs for the driver server and the simulator, run them in separate terminals:

1. **Terminal 1: Start the Drive Server**
   ```bash
   python drive.py
   ```
   *This loads the model, initializes Whisper STT, and listens on port `4567` for connections.*
   
2. **Terminal 2: Start the Simulator**
   ```bash
   python simulator.py
   ```
   *This opens the Pygame simulator, connects to `drive.py`, and begins driving autonomously using inference.*

---

### Option C: Running with the Udacity Beta Simulator
If you want to use the official Udacity Self-Driving Car Simulator rather than the built-in Pygame-based tracker:

1. Start the drive server:
   ```bash
   python drive.py
   ```
2. Open the simulator application at [beta_simulator.exe](file:///c:/Users/athar/OneDrive/Desktop/Nirmaan/term1-simulator-windows/beta_simulator_windows/beta_simulator.exe).
3. Select your screen resolution and graphic settings, click **Play!**.
4. Select a track (e.g., Lake Track) and select **AUTONOMOUS MODE**. The car will immediately start driving using control inputs calculated by `drive.py`.

---

## 🎙️ Voice Commands

Say these into your microphone while the car is driving:

| Say | Effect |
|---|---|
| *"stop"*, *"brake"*, *"halt"* | Stop gradually (throttle = 0.0, uses model to remain on track) |
| *"start"*, *"resume"*, *"auto"* | Keep driving autonomously using the trained model |
| *"left"* | Steer left (3-second duration, safety-assisted to stay on track) |
| *"right"* | Steer right (3-second duration, safety-assisted to stay on track) |

**Pipeline:** Hold **SPACEBAR** to record → Release to stop → **Whisper** (local STT) → action override (with lane-keeping safety check)

---

## 📡 Socket Protocol (Port 4567)

**Receives (`telemetry`):**
```json
{ "image": "<base64 frame>", "speed": "22.5" }
```

**Emits (`steer`):**
```json
{ "steering_angle": "0.12", "throttle": "0.65" }
```

Compatible with `simulator.py` and the **Udacity Self-Driving Car Simulator**.
