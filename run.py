#!/usr/bin/env python3
"""
Launch the llama‑server demo in true head‑less mode.
Optimized for Google Colab notebooks with persistent ngrok tunnels.
"""
import os
import subprocess
import sys
import time
import socket
import json
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------#
#  Utility functions
# ---------------------------------------------------------------------------#


def run(cmd, *, shell=False, cwd=None, env=None, capture=False):
    """Run a command and return its stdout (if capture=True)."""
    if env is None:
        env = os.environ.copy()
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            env=env,
            check=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.strip() if capture else None
    except subprocess.CalledProcessError as exc:
        print(
            f"[ERROR] Command failed: {exc.cmd}\n{exc.stdout or exc.stderr}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def wait_for_service(url, timeout=30, interval=1):
    """Wait for a service to become available at the given URL."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def save_service_info(tunnel_url, llama_pid, streamlit_pid, ngrok_pid):
    """Save service information to a JSON file for later retrieval."""
    info = {
        "tunnel_url": tunnel_url,
        "llama_server_pid": llama_pid,
        "streamlit_pid": streamlit_pid,
        "ngrok_pid": ngrok_pid,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("service_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Also save just the URL to a simple text file
    with open("tunnel_url.txt", "w") as f:
        f.write(tunnel_url)


def get_ngrok_tunnel_url(max_attempts=10, interval=2):
    """Get the tunnel URL from ngrok's local API."""
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=5) as response:
                data = json.loads(response.read())
                if data.get('tunnels'):
                    # Get the public URL (prefer https)
                    for tunnel in data['tunnels']:
                        if tunnel.get('public_url', '').startswith('https'):
                            return tunnel['public_url']
                    # Fallback to first tunnel
                    return data['tunnels'][0]['public_url']
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(interval)
            else:
                raise Exception(f"Failed to get ngrok tunnel URL after {max_attempts} attempts: {e}")
    return None


# ---------------------------------------------------------------------------#
#  Main routine
# ---------------------------------------------------------------------------#


def main():
    # 1. Check prerequisites
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        sys.exit("[ERROR] GITHUB_TOKEN not set")

    NGROK_TOKEN = os.getenv("NGROK_TOKEN")
    if not NGROK_TOKEN:
        sys.exit("[ERROR] NGROK_TOKEN not set")

    # Check for port conflicts
    if is_port_in_use(4040):
        sys.exit("[ERROR] Port 4040 (ngrok API) is already in use")
    if is_port_in_use(8000):
        sys.exit("[ERROR] Port 8000 (llama-server) is already in use")
    if is_port_in_use(8002):
        sys.exit("[ERROR] Port 8002 (Streamlit) is already in use")

    # 2. Download the pre‑built llama‑server binary
    REPO = "ghghang2/llamacpp_t4_v1"
    FILES = ["llama-server"]
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = GITHUB_TOKEN

    for f in FILES:
        run(
            f"gh release download --repo {REPO} --pattern {f}",
            shell=True,
            env=env,
        )
        run(f"chmod +x ./{f}", shell=True)

    # 3. Start llama‑server
    # Keep the file handle open so the subprocess can write to it
    llama_log = Path("llama_server.log").open("w", encoding="utf-8", buffering=1)
    llama_proc = subprocess.Popen(
        [
            "./llama-server",
            "-hf",
            "unsloth/gpt-oss-20b-GGUF:F16",
            "--port",
            "8000",
        ],
        stdout=llama_log,
        stderr=llama_log,
        start_new_session=True,
    )
    
    print("✅ llama-server started (PID: {}), waiting for it to be ready...".format(llama_proc.pid))
    
    # Wait for llama-server to be ready
    if not wait_for_service("http://localhost:8000/health", timeout=240):
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] llama-server failed to start within 60 seconds")
    
    print("✅ llama-server is ready on port 8000")

    # 4. Install required Python packages
    print("📦 Installing Python packages...")
    run("pip install -q streamlit pygithub", shell=True)

    # 5. Start the Streamlit UI
    streamlit_log = Path("streamlit.log").open("w", encoding="utf-8", buffering=1)
    streamlit_proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8002",
            "--server.headless",
            "true",
        ],
        stdout=streamlit_log,
        stderr=streamlit_log,
        start_new_session=True,
    )

    print("✅ Streamlit started (PID: {}), waiting for it to be ready...".format(streamlit_proc.pid))
    
    # Wait for Streamlit to be ready
    if not wait_for_service("http://localhost:8002", timeout=30):
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] Streamlit failed to start within 30 seconds")
    
    print("✅ Streamlit is ready on port 8002")

    # 6. Start ngrok as a background process
    print("🌐 Setting up ngrok tunnel...")
    
    # Create ngrok config file
    ngrok_config = f"""version: 2
authtoken: {NGROK_TOKEN}
tunnels:
  streamlit:
    proto: http
    addr: 8002
"""
    with open("ngrok.yml", "w") as f:
        f.write(ngrok_config)
    
    # Start ngrok in background
    ngrok_log = Path("ngrok.log").open("w", encoding="utf-8", buffering=1)
    ngrok_proc = subprocess.Popen(
        ["ngrok", "start", "--all", "--config", "ngrok.yml", "--log", "stdout"],
        stdout=ngrok_log,
        stderr=ngrok_log,
        start_new_session=True,
    )
    
    print("✅ ngrok started (PID: {}), waiting for tunnel to establish...".format(ngrok_proc.pid))
    
    # Wait for ngrok API to be available
    if not wait_for_service("http://localhost:4040/api/tunnels", timeout=15):
        ngrok_proc.terminate()
        ngrok_log.close()
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] ngrok API did not become available")
    
    # Get tunnel URL from ngrok API
    try:
        tunnel_url = get_ngrok_tunnel_url()
        if not tunnel_url:
            raise Exception("No tunnel URL found")
    except Exception as e:
        print(f"[ERROR] Could not get tunnel URL: {e}")
        print("Check ngrok.log for details:")
        run("tail -20 ngrok.log", shell=True)
        ngrok_proc.terminate()
        ngrok_log.close()
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit(1)
    
    print("✅ ngrok tunnel established")

    # Save service information for later retrieval
    save_service_info(tunnel_url, llama_proc.pid, streamlit_proc.pid, ngrok_proc.pid)

    print("\n" + "="*70)
    print("🎉 ALL SERVICES RUNNING SUCCESSFULLY!")
    print("="*70)
    print(f"🌐 Public URL: {tunnel_url}")
    print(f"🦙 llama-server PID: {llama_proc.pid}")
    print(f"📊 Streamlit PID: {streamlit_proc.pid}")
    print(f"🔌 ngrok PID: {ngrok_proc.pid}")
    print("="*70)
    print("\n📝 Service info saved to: service_info.json")
    print("📝 Tunnel URL saved to: tunnel_url.txt")
    print("📋 Logs available at: llama_server.log, streamlit.log, ngrok.log")
    print("\n💡 To check status later, run: !python launch_demo.py --status")
    print("💡 To get tunnel URL, run: !cat tunnel_url.txt")
    print("💡 To stop services, run: !python launch_demo.py --stop")
    print("\nℹ️  Services will continue running in the background.")
    print("ℹ️  The ngrok tunnel will remain active until stopped.")


def status():
    """Check the status of running services."""
    if not Path("service_info.json").exists():
        print("❌ No service info found. Services may not be running.")
        return
    
    with open("service_info.json", "r") as f:
        info = json.load(f)
    
    print("\n" + "="*70)
    print("SERVICE STATUS")
    print("="*70)
    print(f"Started at: {info['started_at']}")
    print(f"Public URL: {info['tunnel_url']}")
    print(f"llama-server PID: {info['llama_server_pid']}")
    print(f"Streamlit PID: {info['streamlit_pid']}")
    print(f"ngrok PID: {info['ngrok_pid']}")
    print("="*70)
    
    # Check if processes are still running
    for name, pid in [("llama-server", info['llama_server_pid']), 
                       ("Streamlit", info['streamlit_pid']),
                       ("ngrok", info['ngrok_pid'])]:
        try:
            os.kill(pid, 0)  # Check if process exists
            print(f"✅ {name} is running (PID: {pid})")
        except OSError:
            print(f"❌ {name} is NOT running (PID: {pid})")
    
    # Verify tunnel is still active
    print("\n🔍 Checking ngrok tunnel status...")
    try:
        tunnel_url = get_ngrok_tunnel_url(max_attempts=2, interval=1)
        if tunnel_url:
            print(f"✅ Tunnel is active: {tunnel_url}")
        else:
            print("⚠️  Could not verify tunnel status")
    except Exception as e:
        print(f"⚠️  Tunnel check failed: {e}")
    
    print("\n📋 Recent log entries:")
    print("\n--- llama_server.log (last 5 lines) ---")
    if Path("llama_server.log").exists():
        run("tail -5 llama_server.log", shell=True)
    
    print("\n--- streamlit.log (last 5 lines) ---")
    if Path("streamlit.log").exists():
        run("tail -5 streamlit.log", shell=True)
    
    print("\n--- ngrok.log (last 5 lines) ---")
    if Path("ngrok.log").exists():
        run("tail -5 ngrok.log", shell=True)


def stop():
    """Stop all running services."""
    if not Path("service_info.json").exists():
        print("❌ No service info found. Services may not be running.")
        return
    
    with open("service_info.json", "r") as f:
        info = json.load(f)
    
    print("🛑 Stopping services...")
    
    for name, pid in [("llama-server", info['llama_server_pid']), 
                       ("Streamlit", info['streamlit_pid']),
                       ("ngrok", info['ngrok_pid'])]:
        try:
            os.kill(pid, 15)  # SIGTERM
            print(f"✅ Stopped {name} (PID: {pid})")
            time.sleep(0.5)  # Give process time to terminate
        except OSError:
            print(f"⚠️  {name} (PID: {pid}) was not running")
    
    print("\n✅ All services stopped")
    
    # Clean up service info file
    try:
        os.remove("service_info.json")
        os.remove("tunnel_url.txt")
        print("🧹 Cleaned up service info files")
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            status()
        elif sys.argv[1] == "--stop":
            stop()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python launch_demo.py [--status|--stop]")
            sys.exit(1)
    else:
        main()