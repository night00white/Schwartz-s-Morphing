import sys
import subprocess
import os
import site

# 1. Ensure User Site Packages is in Path
# Blender's python sometimes ignores this or has a weird config.
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)
    print(f"[INFO] Added user site-packages to path: {user_site}")

# 2. Dependency Check & Install Function
def install(package):
    print(f"[INFO] Installing {package}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            package, 
            "--only-binary=:all:",
            "--user",
            "--upgrade"
        ])
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package}: {e}")

# 3. Import Check
required = {
    'flask': 'flask', 
    'cv2': 'opencv-python-headless', 
    'mediapipe': 'mediapipe==0.10.9', 
    'numpy': 'numpy==1.26.4',
    'scipy': 'scipy'
}

for module, pip_name in required.items():
    try:
        __import__(module)
        print(f"[CHECK] {module} is available.")
    except ImportError:
        print(f"[WARN] {module} not found. Attempting install...")
        install(pip_name)
        # Try import again
        try:
            __import__(module)
            print(f"[SUCCESS] {module} installed and loaded.")
        except ImportError:
            print(f"[CRITICAL] Could not load {module} after install.")

# 4. Run App
print("[INFO] Starting Flask Server...")
try:
    # We import app here to ensure it uses the modified sys.path
    import app
    # Force run on 0.0.0.0 to avoid binding issues, and port 8080
    app.app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
except Exception as e:
    print(f"[CRITICAL] Server crashed: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
