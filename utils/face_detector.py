from insightface.app import FaceAnalysis
import numpy as np
import torch
import logging

INSIGHTFACE_DETECT_SIZE = 512

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, device="cuda", min_face_size=50, min_face_height=80, 
                 aspect_ratio_range=(0.2, 1.5), detection_threshold=0.5, debug=True):
        """
        Initialize FaceDetector with configurable parameters
        
        Args:
            device: cuda device
            min_face_size: minimum face width
            min_face_height: minimum face height  
            aspect_ratio_range: (min_ratio, max_ratio) for width/height
            detection_threshold: default detection threshold
            debug: whether to print debug information
        """
        self.min_face_size = min_face_size
        self.min_face_height = min_face_height
        self.aspect_ratio_range = aspect_ratio_range
        self.detection_threshold = detection_threshold
        self.debug = debug
        
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root="checkpoints/auxiliary",
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))
        
        if self.debug:
            logger.info(f"FaceDetector initialized with min_face_size={min_face_size}, "
                       f"min_face_height={min_face_height}, aspect_ratio_range={aspect_ratio_range}")

    def __call__(self, frame, threshold=None):
        # Use instance default threshold if none provided
        if threshold is None:
            threshold = self.detection_threshold
            
        f_h, f_w, _ = frame.shape
        
        if self.debug:
            logger.info(f"Processing frame of size: {f_w}x{f_h}, using threshold: {threshold}")

        faces = self.app.get(frame)
        
        if self.debug:
            logger.info(f"InsightFace detected {len(faces)} face(s)")

        get_face_store = None
        max_size = 0
        filtered_faces = []

        if len(faces) == 0:
            if self.debug:
                logger.warning("No faces detected by InsightFace")
            return None, None
        else:
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(np.int_).tolist()
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                # Debug info for each face
                if self.debug:
                    logger.info(f"Face {i}: bbox={bbox}, size={w}x{h}, "
                               f"aspect_ratio={w/h:.2f}, det_score={face.det_score:.3f}")
                
                # Size filtering
                if w < self.min_face_size or h < self.min_face_height:
                    if self.debug:
                        logger.info(f"Face {i} filtered out: too small (min_size={self.min_face_size}, min_height={self.min_face_height})")
                    continue
                
                # Aspect ratio filtering
                aspect_ratio = w / h
                if aspect_ratio > self.aspect_ratio_range[1] or aspect_ratio < self.aspect_ratio_range[0]:
                    if self.debug:
                        logger.info(f"Face {i} filtered out: aspect ratio {aspect_ratio:.2f} not in range {self.aspect_ratio_range}")
                    continue
                
                # Confidence filtering
                if face.det_score < threshold:
                    if self.debug:
                        logger.info(f"Face {i} filtered out: confidence {face.det_score:.3f} < {threshold}")
                    continue
                
                # This face passed all filters
                filtered_faces.append((i, face, w, h))
                size_now = w * h

                if size_now > max_size:
                    max_size = size_now
                    get_face_store = face

        if self.debug:
            logger.info(f"After filtering: {len(filtered_faces)} face(s) remain")

        if get_face_store is None:
            if self.debug:
                logger.warning("No faces passed the filtering criteria")
                if len(faces) > 0:
                    logger.info("Suggestion: try lowering detection threshold or adjusting size/aspect ratio requirements")
            return None, None
        else:
            face = get_face_store
            lmk = np.round(face.landmark_2d_106).astype(np.int_)

            halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)  # lmk[73]

            sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
            halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
            upper_bond = halk_face_coord[1] - halk_face_dist  # *0.94

            x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

            y2 += int((x2 - x1) * 0.1)
            x1 -= int((x2 - x1) * 0.05)
            x2 += int((x2 - x1) * 0.05)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)
            
            if self.debug:
                logger.info(f"Selected face bbox: ({x1}, {y1}, {x2}, {y2})")

            return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
