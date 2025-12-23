
import sys
import os

# Add local site-packages to path just in case
sys.path.append(os.path.join(os.getcwd(), "site-packages"))

print("Attempting normal import...")
try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Solutions attr: {hasattr(mp, 'solutions')}")
    if hasattr(mp, 'solutions'):
        print(f"Face detection: {mp.solutions.face_detection}")
except Exception as e:
    print(f"Normal import failed: {e}")

print("\nAttempting explicit submodule import...")
try:
    import mediapipe.python.solutions as mp_solutions
    print("Explicit import successful")
except Exception as e:
    print(f"Explicit import failed: {e}")
