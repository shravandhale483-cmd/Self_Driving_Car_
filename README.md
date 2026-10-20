# 🚗 Autonomous + Voice-Controlled Self-Driving Car

An end-to-end self-driving car simulation project that combines **Deep Learning, Computer Vision, and Voice Recognition**. The system trains a CNN model for autonomous steering and supports real-time voice commands using OpenAI Whisper for speech-to-text processing.

The project demonstrates the complete workflow: dataset generation, model training, autonomous driving, simulator integration, and voice-command overrides.

## ✨ Key Features

- 🤖 **Autonomous Driving:** Predicts steering commands using a trained CNN model.
- 🧠 **Deep Learning:** Uses an NVIDIA-style CNN architecture for image-based steering prediction.
- 🎙️ **Voice Control:** Supports voice commands through Whisper speech recognition.
- 🛑 **Safety Overrides:** Provides voice commands for stopping, resuming, and directional control.
- 📸 **Dataset Generation:** Generates driving data and records camera frames with vehicle telemetry.
- 🔌 **Socket.IO Communication:** Exchanges driving telemetry and control commands in real time.
- 🕹️ **Simulator Integration:** Supports the built-in Pygame simulator and the compatible Udacity Self-Driving Car Simulator.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| TensorFlow / Keras | CNN model training and inference |
| OpenCV | Image processing, if used by the implementation |
| Pygame | Built-in simulator |
| Socket.IO | Communication between simulator and driving server |
| Whisper | Speech-to-text for voice commands |
| HTML, CSS, JavaScript | Web technologies, if used by the application |

## 📂 Project Structure

```text
Nirmaan/
├── data/
│   ├── IMG/                    # Camera frames
│   └── driving_log.csv         # Driving telemetry
├── generate_dataset.py         # Dataset generation
├── train.py                    # CNN model training
├── drive.py                    # Driving server and voice control
├── simulator.py                # Built-in driving simulator
├── run_autonomous.py           # Automated startup script
├── model.h5                    # Trained model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Nirmaan
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL. If the repository's root folder has a different name, use that folder name instead.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure the installed Python version is compatible with the project's dependencies.

## 📊 Dataset Generation

Generate a dataset using the provided script:

```bash
python generate_dataset.py
```

Alternatively, if supported by the simulator:

1. Run `simulator.py`.
2. Press `M` to enter manual mode.
3. Press `R` to record driving data.

The dataset may include camera frames and telemetry such as steering angle, throttle, brake, and speed.

## 🧠 Train the CNN Model

Run:

```bash
python train.py
```

The training script is expected to produce a trained model named `model.h5`.

If a trained model is already available and compatible with the current environment, you can proceed to autonomous driving without retraining.

## 🤖 Run Autonomous Driving

### Option A: Automated Startup (Recommended)

Run the orchestration script:

```bash
python run_autonomous.py
```

This script is designed to start the driving server, allow initialization, and launch the built-in simulator.

To exit, close the simulator window or use `Ctrl+C` in the terminal.

### Option B: Start Components Manually

**Terminal 1 — Start the driving server**

```bash
python drive.py
```

**Terminal 2 — Start the simulator**

```bash
python simulator.py
```

The simulator should connect to the driving server and send telemetry for model-based steering.

### Option C: Udacity Self-Driving Car Simulator

If the project has been configured for the Udacity simulator:

1. Start the server with `python drive.py`.
2. Open the simulator application.
3. Choose the desired resolution and graphics settings.
4. Select a track.
5. Choose autonomous mode.

The simulator and server must use compatible Socket.IO protocols and settings.

## 🎙️ Voice Commands

The voice-control feature uses microphone input and Whisper speech recognition to interpret commands.

| Voice command | Intended action |
|---|---|
| `stop`, `brake`, `halt` | Stop or reduce throttle |
| `start`, `resume`, `auto` | Resume autonomous driving |
| `left` | Apply a left-steering override |
| `right` | Apply a right-steering override |

### Voice-Control Workflow

1. Hold the **SPACEBAR** to record a voice command.
2. Release the key to finish recording.
3. Whisper converts the speech into text.
4. The recognized command is processed.
5. The driving system applies the corresponding override, subject to the implemented control logic.

Directional overrides are intended to last approximately three seconds, according to the project's command design.

**Safety note:** This is a simulation-oriented project. Voice commands and software-based safety checks are not substitutes for physical emergency stops or validated autonomous-vehicle safety systems.

## 🔌 Socket.IO Communication

The driving server is configured to listen on port `4567`.

### Incoming Telemetry

Example payload:

```json
{
  "image": "<base64-encoded frame>",
  "speed": "22.5"
}
```

### Outgoing Driving Controls

Example payload:

```json
{
  "steering_angle": "0.12",
  "throttle": "0.65"
}
```

These payloads illustrate the expected message structure; the exact event names, value formats, and fields must match the server and simulator implementation.

## 📁 Dependencies

Project dependencies are listed in `requirements.txt`. Depending on the implementation, they may include:

- TensorFlow / Keras
- NumPy
- OpenCV
- Pygame
- Flask-SocketIO or compatible Socket.IO libraries
- Whisper and its required audio-processing dependencies

Install the versions specified by the project whenever possible.

## 🔮 Future Improvements

- Improve model performance on different tracks and conditions.
- Add additional voice commands and configurable control durations.
- Improve telemetry visualization and debugging.
- Add model evaluation metrics and training-history plots.
- Implement stronger command validation and emergency-stop handling.
- Package the application for easier setup and deployment.

## 👨‍💻 Author

Developed as a machine-learning and autonomous-driving simulation project.

## 📜 Disclaimer

This project is intended for educational and simulation purposes only. It has not been validated for use in real vehicles or public-road driving.

---

⭐ If you find this project interesting, consider starring the repository.
