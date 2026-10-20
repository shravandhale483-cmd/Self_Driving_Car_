import subprocess
import time
import sys
import os
import threading

def stream_stdout(pipe, prefix):
    """Streams lines from a subprocess pipe to console with a prefix."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                print(f"{prefix} {line.strip()}")
    except Exception as e:
        print(f"[Run Manager] Error streaming logs: {e}")

def main():
    print("=" * 65)
    print("🚀 Starting Autonomous Self-Driving Car Simulator & Driver...")
    print("=" * 65)
    
    # Locate scripts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    drive_path = os.path.join(base_dir, "drive.py")
    simulator_path = os.path.join(base_dir, "simulator.py")
    
    # 1. Start the drive server
    print(f"[Run Manager] Launching drive server: python drive.py")
    drive_proc = subprocess.Popen(
        [sys.executable, drive_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="ignore"
    )
    
    # Read output to print until drive server is listening or fails
    ready = False
    start_time = time.time()
    startup_timeout = 45  # Whisper + model loading can take a moment
    
    print("[Run Manager] Waiting for drive server initialization...")
    try:
        while True:
            # Read a line from the server stdout
            line = drive_proc.stdout.readline()
            if not line:
                break
            print(f"[Drive Server] {line.strip()}")
            
            # Check if socket server is listening
            if "ws://localhost:" in line or "Autonomous + Voice-Controlled Driver" in line:
                ready = True
                break
            
            # Check for immediate crash/exit
            if drive_proc.poll() is not None:
                print("[Run Manager] [ERROR] Drive server exited unexpectedly during startup.")
                sys.exit(1)
                
            if time.time() - start_time > startup_timeout:
                print("[Run Manager] [WARN] Startup timeout reached. Proceeding to launch simulator...")
                break
    except Exception as e:
        print(f"[Run Manager] Error monitoring drive server startup: {e}")
    
    # 2. Launch the simulator (which opens Pygame in AUTONOMOUS mode)
    print(f"\n[Run Manager] Launching Pygame Simulator: python simulator.py")
    sim_proc = subprocess.Popen(
        [sys.executable, simulator_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="ignore"
    )
    
    # Start background threads to stream output from both processes
    t_drive = threading.Thread(target=stream_stdout, args=(drive_proc.stdout, "[Drive Server]"), daemon=True)
    t_sim = threading.Thread(target=stream_stdout, args=(sim_proc.stdout, "[Simulator]   "), daemon=True)
    t_drive.start()
    t_sim.start()
    
    print("\n" + "=" * 65)
    print("🎉 RUNNING IN FULLY AUTONOMOUS MODE!")
    print("   - Close the Pygame window to stop everything.")
    print("   - Or press Ctrl+C in this terminal window.")
    print("=" * 65 + "\n")
    
    # 3. Keep running until simulator terminates or user interrupts
    try:
        while sim_proc.poll() is None:
            # Check if drive server has died
            if drive_proc.poll() is not None:
                print("[Run Manager] [WARN] Drive server stopped running.")
                break
            time.sleep(0.5)
        
        if sim_proc.poll() is not None:
            print("\n[Run Manager] Simulator window closed.")
            
    except KeyboardInterrupt:
        print("\n[Run Manager] KeyboardInterrupt received. Terminating processes...")
        
    finally:
        # Cleanup processes
        print("[Run Manager] Stopping simulator...")
        if sim_proc.poll() is None:
            sim_proc.terminate()
            try:
                sim_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sim_proc.kill()
                
        print("[Run Manager] Stopping drive server...")
        if drive_proc.poll() is None:
            drive_proc.terminate()
            try:
                drive_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                drive_proc.kill()
                
    print("[Run Manager] All processes terminated successfully. Exiting.")

if __name__ == "__main__":
    main()
