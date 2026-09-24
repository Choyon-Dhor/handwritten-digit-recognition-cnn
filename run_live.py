import os
import sys

# Force UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pyngrok import ngrok

# Configure auth token
NGROK_AUTH_TOKEN = "36TWYDTKQZ9tpUQXvwn72gFIs2u_4JXmMNUWDXfg6jBR6Zr1w"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

print("Starting Ngrok Public Live Tunnel on Port 5000...")
try:
    public_tunnel = ngrok.connect(5000)
    public_url = public_tunnel.public_url
except Exception as e:
    print(f"Ngrok tunnel notice: {e}")
    public_url = "Check tunnel status"

print("\n" + "=" * 65)
print("  HANDWRITTEN DIGIT RECOGNITION - LIVE DEPLOYMENT")
print("=" * 65)
print(f"  >> PUBLIC LIVE LINK:  {public_url}")
print(f"  >> LOCAL URL:        http://127.0.0.1:5000")
print("=" * 65)
print("  Server is active and accepting requests...\n")

# Import and start Flask application
from app import app
app.run(host="0.0.0.0", port=5000, debug=False)
