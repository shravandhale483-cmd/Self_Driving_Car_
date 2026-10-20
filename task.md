## Voice Control Implementation Tasks

- [x] Read and understand drive.py and requirements.txt
- [x] Update requirements.txt with new dependencies
- [x] Rewrite drive.py with Whisper + Groq voice control integration
- [x] Install new dependencies (groq 1.6.0 installed; whisper/sounddevice/scipy already present)
- [x] Create groq_api_key.txt template
- [x] Syntax check passed — drive.py compiles OK
- [x] All imports verified (whisper, groq, sounddevice, scipy)
- [x] Re-implement keyboard listener using Windows user32 ctypes to bypass eventlet/pynput freeze
- [x] Replaced all Unicode emojis/arrows to resolve CP1252 print encoding crashes
- [x] Simplified command mapping to `brake` and `reset` (toggle mode)
- [x] Implemented immediate speed drop to 0.0 in the simulator upon receiving a `brake` command
