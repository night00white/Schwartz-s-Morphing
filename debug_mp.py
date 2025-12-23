import mediapipe as mp
print(dir(mp.solutions))
try:
    print(mp.solutions.face_detection)
    print("Face Detection OK")
except AttributeError as e:
    print(f"Face Detection Error: {e}")
