import os
import sys

try:
    from waitress import serve
except ImportError:
    print("Waitress is not installed. Please run: pip install waitress")
    sys.exit(1)

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[DEPLOY] Starting production server on port {port} using Waitress...")
    print(f"[DEPLOY] Application is now live. Access it at http://localhost:{port}")
    serve(app, host="0.0.0.0", port=port)
