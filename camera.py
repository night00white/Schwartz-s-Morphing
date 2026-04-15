import cv2
import mediapipe as mp
try:
    import mediapipe.python.solutions
except ImportError:
    pass
    
import numpy as np
from scipy.spatial import Delaunay
import time
import os

CONF_THRESH = 0.5
CROP_MARGIN = 0.25
CROP_SIZE = 400

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

class FaceMorpher:
    def __init__(self):
        self.fd = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.fm = mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.fm_static = mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        self.sequence_data = [] # List of tuples: (ref_img, ref_pts)
        self.seq_index = 0
        self.transition_start = 0.0
        self.duration = 4.0 # 4 seconds per transition phase
        self.ref_loaded = False
        
        self.captured_frames = []
        self.sequence_complete = False
        self.grid_image = None
        
        if os.path.exists("default.jpg"):
            self.set_reference("default.jpg")

    def load_reference_data(self, path):
        img = cv2.imread(path)
        if img is None: return None, None
        img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
        res = self.fm_static.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: return None, None
        pts = res.multi_face_landmarks[0].landmark
        pts = np.array([[int(p.x * CROP_SIZE), int(p.y * CROP_SIZE)] for p in pts], dtype=np.int32)
        return img, pts

    def set_reference(self, path):
        print(f"Loading sequence starting with {path}")
        self.sequence_data = []
        self.captured_frames = []
        self.sequence_complete = False
        self.grid_image = None
        
        img1, pts1 = self.load_reference_data(path)
        if img1 is not None:
            self.sequence_data.append((img1, pts1))
            
            # If Monalisa, add Davici sequence
            if "MONA LISA" in path.upper():
                da_vinci_path = path.replace("MONA LISA.png", "DAVICI.png").replace("mona_lisa", "Davici")
                img2, pts2 = self.load_reference_data(da_vinci_path)
                if img2 is not None:
                    self.sequence_data.append((img2, pts2))
            # If Davici, add Monalisa sequence
            elif "DAVICI" in path.upper():
                mona_lisa_path = path.replace("DAVICI.png", "MONA LISA.png").replace("Davici", "mona_lisa")
                img2, pts2 = self.load_reference_data(mona_lisa_path)
                if img2 is not None:
                    self.sequence_data.append((img2, pts2))
                    
        self.seq_index = 0
        self.transition_start = time.time()
        self.ref_loaded = len(self.sequence_data) > 0
        return self.ref_loaded

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

    def warp_ref_to_detected(self, det_bgr, det_pts, ref_img, ref_pts):
        H, W = det_bgr.shape[:2]
        
        ref_all = self.add_boundary_points(ref_pts, CROP_SIZE, CROP_SIZE)
        det_all = self.add_boundary_points(det_pts, H, W)

        tri = Delaunay(ref_all)
        out = det_bgr.copy().astype(np.float32)

        for ids in tri.simplices:
            t_src = ref_all[ids]
            t_out = det_all[ids]
            self.morph_triangle(ref_img, out, t_src, t_out)

        return out

    def process_frame(self, frame):
        if frame is None:
            return None
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        if best_xywh:
            x, y, ww, hh = best_xywh
            
            if self.ref_loaded:
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
                        import math
                        if self.sequence_complete and self.grid_image is not None:
                            # Instead of turning the video into the grid, we just continue emitting
                            # the 100% frozen morphed frame. The grid is requested separately.
                            pass

                        elapsed = time.time() - self.transition_start
                        
                        # Handle Stage Progression
                        if self.seq_index == 0 and len(self.sequence_data) > 1:
                            if elapsed > self.duration:
                                self.seq_index = 1
                                self.transition_start = time.time()
                                elapsed = 0.0
                        
                        if self.seq_index == 0:
                            # Stage 0: Camera -> First Target (Clamp at 1.0)
                            if len(self.sequence_data) > 1:
                                progress = min(1.0, elapsed / self.duration)
                            else:
                                # For single target: just go 0 to 1 and stop
                                progress = min(1.0, elapsed / self.duration)
                                
                            img_B, pts_B = self.sequence_data[0]
                            morphed_B = self.warp_ref_to_detected(crop, crop_pts, img_B, pts_B)
                            morphed_A = crop.astype(np.float32)
                            blended = cv2.addWeighted(morphed_A, 1.0 - progress, morphed_B, progress, 0.0)
                        else:
                            # Stage 1: Target 1 -> Target 2 (Clamp at 1.0)
                            progress = min(1.0, elapsed / self.duration)
                            img_A, pts_A = self.sequence_data[0]
                            img_B, pts_B = self.sequence_data[1]
                            morphed_A = self.warp_ref_to_detected(crop, crop_pts, img_A, pts_A)
                            morphed_B = self.warp_ref_to_detected(crop, crop_pts, img_B, pts_B)
                            blended = cv2.addWeighted(morphed_A, 1.0 - progress, morphed_B, progress, 0.0)

                        blended = np.clip(blended, 0, 255).astype(np.uint8)
                        
                        # Calculate global progress 0 to 1
                        total_stages = len(self.sequence_data)
                        global_progress = (self.seq_index + progress) / total_stages
                        
                        # Capture frames for grid
                        target_frames = 9
                        expected_captures = min(target_frames, int(global_progress * target_frames) + 1)
                        while len(self.captured_frames) < expected_captures and len(self.captured_frames) < target_frames:
                            snap = blended.copy()
                            cv2.putText(snap, f"{int(global_progress * 100)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                            cv2.putText(snap, f"{int(global_progress * 100)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            self.captured_frames.append(snap)
                            
                        # If sequence ends, create grid!
                        if progress >= 1.0 and self.seq_index == total_stages - 1:
                            while len(self.captured_frames) < 9:
                                self.captured_frames.append(blended.copy())
                                
                            row1 = np.hstack(self.captured_frames[0:3])
                            row2 = np.hstack(self.captured_frames[3:6])
                            row3 = np.hstack(self.captured_frames[6:9])
                            grid_img = np.vstack([row1, row2, row3])
                            self.grid_image = grid_img
                            self.sequence_complete = True
                            
                        morphed_back = cv2.resize(blended, (base_crop.shape[1], base_crop.shape[0]))
                        
                        # Create an edge-blending mask for lower opacity along the edges
                        cw, ch = base_crop.shape[1], base_crop.shape[0]
                        X, Y = np.meshgrid(np.linspace(-1, 1, cw), np.linspace(-1, 1, ch))
                        r = np.sqrt(X**2 + Y**2)
                        mask = 1.0 - np.clip((r - 0.5) / 0.5, 0.0, 1.0)
                        mask = mask * mask * (3.0 - 2.0 * mask) # Smoothstep
                        mask = mask[..., np.newaxis]
                        
                        original_bg = base_crop.astype(np.float32)
                        morphed_f = morphed_back.astype(np.float32)
                        blended_bg = (morphed_f * mask + original_bg * (1.0 - mask)).astype(np.uint8)
                        
                        # Blend the morphed face into the original frame backwards
                        frame[y0:y1, x0:x1] = blended_bg
                        
                        # Add percentage text
                        text = f"{int(global_progress * 100)}%"
                        cv2.putText(frame, text, (x0, max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
                        cv2.putText(frame, text, (x0, max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return frame
