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

# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #
def run(cmd, *, shell=False, cwd=None, env=None, capture=False):
    """Run a command and optionally capture its output."""
    env = env or os.environ.copy()
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

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def wait_for_service(url, timeout=30, interval=1):
    """Wait for a service to respond with HTTP 200."""
    for _ in range(int(timeout / interval)):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False

def save_service_info(tunnel_url, llama_pid, streamlit_pid, ngrok_pid):
    """Persist service info for later queries."""
    info = {
        "tunnel_url": tunnel_url,
        "llama_server_pid": llama_pid,
        "streamlit_pid": streamlit_pid,
        "ngrok_pid": ngrok_pid,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path("service_info.json").write_text(json.dumps(info, indent=2))
    Path("tunnel_url.txt").write_text(tunnel_url)

# --------------------------------------------------------------------------- #
#  Main routine
# --------------------------------------------------------------------------- #
def main():
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    NGROK_TOKEN = os.getenv("NGROK_TOKEN")
    if not GITHUB_TOKEN or not NGROK_TOKEN:
        sys.exit("[ERROR] GITHUB_TOKEN and NGROK_TOKEN must be set")

    for port in (4040, 8000, 8002):
        if is_port_in_use(port):
            sys.exit(f"[ERROR] Port {port} is already in use")

    # 1️⃣  Download the pre‑built llama‑server binary
    REPO = "ghghang2/llamacpp_t4_v1"
    run(f"gh release download --repo {REPO} --pattern llama-server", shell=True, env={"GITHUB_TOKEN": GITHUB_TOKEN})
    run("chmod +x ./llama-server", shell=True)

    # 2️⃣  Start llama‑server
    llama_log = Path("llama_server.log").open("w", encoding="utf-8", buffering=1)
    llama_proc = subprocess.Popen(
        ["./llama-server", "-hf", "unsloth/gpt-oss-20b-GGUF:F16", "--port", "8000"],
        stdout=llama_log,
        stderr=llama_log,
        start_new_session=True,
    )
    print(f"✅ llama-server started (PID: {llama_proc.pid}), waiting for ready…")
    if not wait_for_service("http://localhost:8000/health", timeout=240):
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] llama-server failed to start")

    print("✅ llama-server is ready on port 8000")

    # 3️⃣  Install required Python packages
    print("📦 Installing Python packages…")
    run("pip install -q streamlit pygithub pyngrok", shell=True)

    # 4️⃣  Start Streamlit UI
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
    print(f"✅ Streamlit started (PID: {streamlit_proc.pid}), waiting for ready…")
    if not wait_for_service("http://localhost:8002", timeout=30):
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] Streamlit failed to start")

    print("✅ Streamlit is ready on port 8002")

    # 5️⃣  Start ngrok
    print("🌐 Setting up ngrok tunnel…")
    ngrok_config = f"""version: 2
authtoken: {NGROK_TOKEN}
tunnels:
  streamlit:
    proto: http
    addr: 8002
"""
    Path("ngrok.yml").write_text(ngrok_config)

    ngrok_log = Path("ngrok.log").open("w", encoding="utf-8", buffering=1)
    ngrok_proc = subprocess.Popen(
        ["ngrok", "start", "--all", "--config", "ngrok.yml", "--log", "stdout"],
        stdout=ngrok_log,
        stderr=ngrok_log,
        start_new_session=True,
    )
    print(f"✅ ngrok started (PID: {ngrok_proc.pid}), waiting for tunnel…")
    if not wait_for_service("http://localhost:4040/api/tunnels", timeout=15):
        ngrok_proc.terminate()
        ngrok_log.close()
        streamlit_proc.terminate()
        streamlit_log.close()
        llama_proc.terminate()
        llama_log.close()
        sys.exit("[ERROR] ngrok API did not become available")

    # Grab the public URL
    try:
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=5) as r:
            tunnels = json.loads(r.read())
            tunnel_url = next(
                (t["public_url"] for t in tunnels["tunnels"] if t["public_url"].startswith("https")),
                tunnels["tunnels"][0]["public_url"],
            )
    except Exception as exc:
        print(f"[ERROR] Could not get tunnel URL: {exc}")
        sys.exit(1)

    print("✅ ngrok tunnel established")

    # Persist service info
    save_service_info(tunnel_url, llama_proc.pid, streamlit_proc.pid, ngrok_proc.pid)

    print("\n" + "=" * 70)
    print("🎉 ALL SERVICES RUNNING SUCCESSFULLY!")
    print("=" * 70)
    print(f"🌐 Public URL: {tunnel_url}")
    print(f"🦙 llama-server PID: {llama_proc.pid}")
    print(f"📊 Streamlit PID: {streamlit_proc.pid}")
    print(f"🔌 ngrok PID: {ngrok_proc.pid}")
    print("=" * 70)
    print("\n📝 Service info saved to: service_info.json")
    print("📝 Tunnel URL saved to: tunnel_url.txt")

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