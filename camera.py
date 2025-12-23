import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay
import time
import os

# Configuration
CONF_THRESH = 0.5
CROP_MARGIN = 0.25
CROP_SIZE = 400

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

class FaceMorpher:
    def __init__(self):
        self.fd = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        # Refined landmarks=False for speed, or True for iris if needed (original script used False)
        self.fm = mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.ref_img = None
        self.ref_pts = None
        self.ref_loaded = False
        
        # Helper for static mesh
        self.fm_static = mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        # Try to load default reference
        if os.path.exists("default.jpg"):
            print("Loading default reference...")
            self.set_reference("default.jpg")

    def set_reference(self, path):
        print(f"Loading reference from {path}")
        img = cv2.imread(path)
        if img is None:
            print("Failed to load reference image.")
            self.ref_loaded = False
            return False
        
        self.ref_img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
        
        # Get landmarks for reference
        res = self.fm_static.process(cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            print("No face found in reference image.")
            self.ref_loaded = False
            return False
            
        pts = res.multi_face_landmarks[0].landmark
        self.ref_pts = np.array([[int(p.x * CROP_SIZE), int(p.y * CROP_SIZE)] for p in pts], dtype=np.int32)
        self.ref_loaded = True
        return True

    def get_facemesh_points(self, image_bgr):
        h, w = image_bgr.shape[:2]
        res = self.fm.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        pts = res.multi_face_landmarks[0].landmark
        return np.array([[int(p.x * w), int(p.y * h)] for p in pts], dtype=np.int32)

    def morph_triangle(self, img_src, out_img, t_src, t_out):
        r_src = cv2.boundingRect(np.float32([t_src]))
        r_out = cv2.boundingRect(np.float32([t_out]))

        t_src_rect = np.float32([[p[0] - r_src[0], p[1] - r_src[1]] for p in t_src])
        t_out_rect = np.float32([[p[0] - r_out[0], p[1] - r_out[1]] for p in t_out])

        if r_src[2] <= 0 or r_src[3] <= 0 or r_out[2] <= 0 or r_out[3] <= 0:
            return

        src_rect = img_src[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]
        M = cv2.getAffineTransform(t_src_rect, t_out_rect)
        warped = cv2.warpAffine(src_rect, M, (r_out[2], r_out[3]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((r_out[3], r_out[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_out_rect), (1.0, 1.0, 1.0), 16, 0)

        # Output might be out of bounds if not careful, but usually okay with bounding rects
        out_slice = out_img[r_out[1]:r_out[1] + r_out[3], r_out[0]:r_out[0] + r_out[2]]
        if out_slice.shape[:2] == warped.shape[:2]:
             out_slice[:] = out_slice * (1 - mask) + warped * mask

    def add_boundary_points(self, points, H, W):
        boundary = np.array([
            (0, 0), (W // 2, 0), (W - 1, 0),
            (0, H // 2), (W - 1, H // 2),
            (0, H - 1), (W // 2, H - 1), (W - 1, H - 1)
        ], dtype=np.int32)
        return np.vstack([points, boundary])

    def warp_ref_to_detected(self, det_bgr, det_pts):
        if not self.ref_loaded:
            return det_bgr
            
        H, W = det_bgr.shape[:2]
        # Resize ref to match dest height/width? Or just warp mapping. 
        # The Original script resizes ref to crop size, and crop is also crop size.
        # But here det_bgr is the crop.
        
        ref_all = self.add_boundary_points(self.ref_pts, CROP_SIZE, CROP_SIZE) # Ref is alwasy CROP_SIZE
        det_all = self.add_boundary_points(det_pts, H, W)

        tri = Delaunay(ref_all)
        out = det_bgr.copy().astype(np.float32)

        for ids in tri.simplices:
            t_src = ref_all[ids]
            t_out = det_all[ids]
            self.morph_triangle(self.ref_img, out, t_src, t_out)

        return np.clip(out, 0, 255).astype(np.uint8)

    def process_frame(self, frame):
        if frame is None:
            return None
        
        # Mirror
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect
        det_res = self.fd.process(rgb)
        best_xywh = None
        best_score = 0.0

        if det_res.detections:
            for d in det_res.detections:
                s = d.score[0] if d.score else 0.0
                bb = d.location_data.relative_bounding_box
                x, y = int(bb.xmin * w), int(bb.ymin * h)
                ww, hh = int(bb.width * w), int(bb.height * h)
                if s > best_score:
                    best_xywh = (x, y, ww, hh)
                    best_score = s

        # Draw box and info
        if best_xywh:
            x, y, ww, hh = best_xywh
            # Debug info removed for cleaner feed
            
            # If we have a reference, let's morph!
            if self.ref_loaded:
                # Crop
                side = int(max(ww, hh) * (1 + 2 * CROP_MARGIN))
                cx, cy = x + ww // 2, y + hh // 2
                x0 = max(0, cx - side // 2)
                y0 = max(0, cy - side // 2)
                x1 = min(w - 1, cx + side // 2)
                y1 = min(h - 1, cy + side // 2)
                
                if x1 > x0 and y1 > y0:
                    base_crop = frame[y0:y1, x0:x1].copy()
                    crop = cv2.resize(base_crop, (CROP_SIZE, CROP_SIZE))
                    
                    crop_pts = self.get_facemesh_points(crop)
                    if crop_pts is not None:
                        morphed = self.warp_ref_to_detected(crop, crop_pts)
                        
                        # Blend back
                        morphed_back = cv2.resize(morphed, (base_crop.shape[1], base_crop.shape[0]))
                        alpha = 0.7
                        blended = cv2.addWeighted(base_crop, 1.0 - alpha, morphed_back, alpha, 0.0)
                        
                        # Place back on frame
                        frame[y0:y1, x0:x1] = blended

        return frame
