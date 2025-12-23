import sys
import subprocess
import os

print(f"Python Executable: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")

# Try to look for pip
try:
    import pip
    print(f"Pip found: {pip.__version__}")
except ImportError:
    print("Pip NOT found in imports.")

cmd = [
    sys.executable, "-m", "pip", "install", 
    "flask", "opencv-python-headless", "mediapipe==0.10.9", "scipy", "numpy==1.26.4",
    "--only-binary=:all:",
    "--target", "site-packages",
    "--no-warn-script-location",
    "--upgrade",
    "--verbose" # Get detailed logs
]

print(f"Running command: {' '.join(cmd)}")

try:
    # ensuring site-packages exists
    if not os.path.exists("site-packages"):
        os.makedirs("site-packages")
        
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True
    )
    
    print("RETURN CODE:", result.returncode)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("SUCCESS: Dependencies installed.")
    else:
        print("FAILURE: Installation failed.")
        
except Exception as e:
    print(f"EXCEPTION: {e}")
