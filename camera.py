import cv2
import numpy as np
from scipy.spatial import Delaunay
import time
import os

from mediapipe.python.solutions import face_detection as mp_face
from mediapipe.python.solutions import face_mesh as mp_mesh

CONF_THRESH = 0.5
CROP_MARGIN = 0.25
CROP_SIZE = 400


class FaceMorpher:
    def __init__(self):
        self.fd = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        self.fm = mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.fm_static = mp_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1
        )

        self.sequence_data = []
        self.seq_index = 0
        self.transition_start = 0.0
        self.duration = 4.0
        self.ref_loaded = False

        self.captured_frames = []
        self.sequence_complete = False
        self.grid_image = None

        if os.path.exists("default.jpg"):
            self.set_reference("default.jpg")

    def load_reference_data(self, path):
        img = cv2.imread(path)
        if img is None:
            return None, None

        img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
        res = self.fm_static.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None

        pts = res.multi_face_landmarks[0].landmark
        pts = np.array(
            [[int(p.x * CROP_SIZE), int(p.y * CROP_SIZE)] for p in pts],
            dtype=np.int32
        )
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
        return np.array(
            [[int(p.x * w), int(p.y * h)] for p in pts],
            dtype=np.int32
        )

    def morph_triangle(self, img_src, out_img, t_src, t_out):
        r_src = cv2.boundingRect(np.float32([t_src]))
        r_out = cv2.boundingRect(np.float32([t_out]))

        t_src_rect = np.float32(
            [[p[0] - r_src[0], p[1] - r_src[1]] for p in t_src]
        )
        t_out_rect = np.float32(
            [[p[0] - r_out[0], p[1] - r_out[1]] for p in t_out]
        )

        if r_src[2] <= 0 or r_src[3] <= 0 or r_out[2] <= 0 or r_out[3] <= 0:
            return

        src_rect = img_src[
            r_src[1]:r_src[1] + r_src[3],
            r_src[0]:r_src[0] + r_src[2]
        ]

        M = cv2.getAffineTransform(t_src_rect, t_out_rect)
        warped = cv2.warpAffine(
            src_rect,
            M,
            (r_out[2], r_out[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        mask = np.zeros((r_out[3], r_out[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_out_rect), (1, 1, 1))

        out_slice = out_img[
            r_out[1]:r_out[1] + r_out[3],
            r_out[0]:r_out[0] + r_out[2]
        ]

        if out_slice.shape[:2] == warped.shape[:2]:
            out_slice[:] = out_slice * (1 - mask) + warped * mask

    def warp_ref_to_detected(self, det_bgr, det_pts, ref_img, ref_pts):
        H, W = det_bgr.shape[:2]

        tri = Delaunay(ref_pts)
        out = det_bgr.copy().astype(np.float32)

        for ids in tri.simplices:
            t_src = ref_pts[ids]
            t_out = det_pts[ids]
            self.morph_triangle(ref_img, out, t_src, t_out)

        return out

    def process_frame(self, frame):
        if frame is None:
            return None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        det_res = self.fd.process(rgb)
        if not det_res.detections:
            return frame

        d = det_res.detections[0]
        bb = d.location_data.relative_bounding_box

        x = int(bb.xmin * w)
        y = int(bb.ymin * h)
        ww = int(bb.width * w)
        hh = int(bb.height * h)

        side = int(max(ww, hh) * (1 + 2 * CROP_MARGIN))
        cx, cy = x + ww // 2, y + hh // 2

        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, cx + side // 2)
        y1 = min(h, cy + side // 2)

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return frame

        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        pts = self.get_facemesh_points(crop)

        if pts is None or not self.sequence_data:
            return frame

        ref_img, ref_pts = self.sequence_data[0]
        morphed = self.warp_ref_to_detected(crop, pts, ref_img, ref_pts)
        morphed = np.clip(morphed, 0, 255).astype(np.uint8)

        frame[y0:y1, x0:x1] = cv2.resize(
            morphed,
            (x1 - x0, y1 - y0)
        )

        return frame
