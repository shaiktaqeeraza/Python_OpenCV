"""
camera_ai_project.py
=====================
Real-time webcam recognition in a SINGLE FILE — no model weights to download.

Detects:
    - People / objects  -> OpenCV's built-in HOG people detector (bundled with opencv-python)
    - Faces              -> OpenCV's built-in Haar Cascade (bundled with opencv-python)
    - Hand gestures      -> MediaPipe Hands (models bundled inside the mediapipe package)

INSTALL (one time):
    pip install opencv-python mediapipe numpy

RUN:
    python camera_ai_project.py

CONTROLS (video window must be focused):
    o = toggle object/person detection
    f = toggle face detection
    h = toggle hand gesture recognition
    q / ESC = quit
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False


# ----------------------------------------------------------------------------
# Drawing / FPS helpers
# ----------------------------------------------------------------------------

def draw_label(frame, text, x, y, color=(0, 255, 0), font_scale=0.6, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(y, th + baseline + 2)
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


class FPSCounter:
    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = None

    def tick(self):
        now = time.time()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * inst
        self._last = now
        return self.fps


# ----------------------------------------------------------------------------
# Object (person) detector — OpenCV built-in HOG, no download needed
# ----------------------------------------------------------------------------

class ObjectDetector:
    """
    Detects people using OpenCV's built-in HOG + Linear SVM detector.
    Fully bundled with opencv-python — nothing to download.
    (Note: this detects PEOPLE specifically, not arbitrary object classes —
    that's the tradeoff for having zero external downloads.)
    """

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        # Downscale for speed, then rescale boxes back up
        h, w = frame.shape[:2]
        scale = 640 / w if w > 640 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame

        boxes, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        results = []
        for (x, y, bw, bh), conf in zip(boxes, weights):
            x1, y1 = int(x / scale), int(y / scale)
            x2, y2 = int((x + bw) / scale), int((y + bh) / scale)
            results.append(("person", float(conf), (x1, y1, x2, y2)))
        return results

    def draw(self, frame, detections):
        for name, conf, (x1, y1, x2, y2) in detections:
            draw_box(frame, x1, y1, x2, y2, color=(0, 200, 0))
            draw_label(frame, f"{name} {conf:.2f}", x1, y1, color=(0, 200, 0))
        return frame


# ----------------------------------------------------------------------------
# Face detector — OpenCV built-in Haar Cascade, no download needed
# ----------------------------------------------------------------------------

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Could not load bundled Haar cascade file from OpenCV install.")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return [(x, y, x + fw, y + fh) for (x, y, fw, fh) in faces]

    def draw(self, frame, faces):
        for (x1, y1, x2, y2) in faces:
            draw_box(frame, x1, y1, x2, y2, color=(255, 200, 0))
            draw_label(frame, "Face", x1, y1, color=(255, 200, 0))
        return frame


# ----------------------------------------------------------------------------
# Hand gesture recognizer — MediaPipe Hands, models bundled in the pip package
# ----------------------------------------------------------------------------

FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


class HandGestureRecognizer:
    def __init__(self, max_hands=2):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def _finger_extended(self, lm, finger):
        tip = lm.landmark[FINGER_TIPS[finger]]
        pip = lm.landmark[FINGER_PIPS[finger]]
        if finger == "thumb":
            wrist = lm.landmark[0]
            return abs(tip.x - wrist.x) > abs(pip.x - wrist.x) * 1.1
        return tip.y < pip.y

    def _classify(self, lm):
        ext = {f: self._finger_extended(lm, f) for f in FINGER_TIPS}
        count = sum(ext.values())
        thumb, index, middle, ring, pinky = (
            ext["thumb"], ext["index"], ext["middle"], ext["ring"], ext["pinky"]
        )

        thumb_tip, index_tip = lm.landmark[4], lm.landmark[8]
        dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        if dist < 0.05 and middle and ring and pinky:
            return "OK"
        if count == 0:
            return "Fist"
        if count == 5:
            return "Open Palm"
        if thumb and not any([index, middle, ring, pinky]):
            return "Thumbs Up"
        if index and middle and not ring and not pinky:
            return "Peace / Victory"
        if index and not any([middle, ring, pinky, thumb]):
            return "Pointing"
        if thumb and pinky and not any([index, middle, ring]):
            return "Call Me"
        return f"{count} Fingers"

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        h, w = frame.shape[:2]
        gestures = []
        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or [None] * len(results.multi_hand_landmarks)
            for lm, handed in zip(results.multi_hand_landmarks, handedness_list):
                label = self._classify(lm)
                side = handed.classification[0].label if handed else "Hand"
                wrist = lm.landmark[0]
                gestures.append((label, side, (int(wrist.x * w), int(wrist.y * h))))
        return results, gestures

    def draw(self, frame, results, gestures):
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        for label, side, (px, py) in gestures:
            draw_label(frame, f"{side}: {label}", px, py, color=(0, 200, 255))
        return frame

    def close(self):
        self.hands.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time camera object/face/hand-gesture recognition (single file)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-objects", action="store_true")
    p.add_argument("--no-faces", action="store_true")
    p.add_argument("--no-hands", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}.")
        sys.exit(1)

    object_detector = ObjectDetector()
    face_detector = FaceDetector()

    hand_recognizer = None
    if _HAS_MEDIAPIPE:
        hand_recognizer = HandGestureRecognizer()
    else:
        print("[WARN] mediapipe not installed — hand gesture recognition disabled. "
              "Run: pip install mediapipe")

    state = {
        "objects": not args.no_objects,
        "faces": not args.no_faces,
        "hands": (not args.no_hands) and hand_recognizer is not None,
    }

    fps_counter = FPSCounter()
    print("Controls: 'o' objects | 'f' faces | 'h' hands | 'q'/ESC quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)

            if state["objects"]:
                dets = object_detector.detect(frame)
                object_detector.draw(frame, dets)

            if state["faces"]:
                faces = face_detector.detect(frame)
                face_detector.draw(frame, faces)

            if state["hands"] and hand_recognizer is not None:
                results, gestures = hand_recognizer.process(frame)
                hand_recognizer.draw(frame, results, gestures)

            fps = fps_counter.tick()
            draw_label(frame, f"FPS: {fps:.1f}", 10, 30, color=(0, 255, 0))
            status = (
                f"[O]bjects:{'ON' if state['objects'] else 'off'}  "
                f"[F]aces:{'ON' if state['faces'] else 'off'}  "
                f"[H]ands:{'ON' if state['hands'] else 'off'}"
            )
            draw_label(frame, status, 10, frame.shape[0] - 10, color=(200, 200, 200))

            cv2.imshow("Real-Time Recognition (o/f/h toggle, q quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("o"):
                state["objects"] = not state["objects"]
            elif key == ord("f"):
                state["faces"] = not state["faces"]
            elif key == ord("h") and hand_recognizer is not None:
                state["hands"] = not state["hands"]

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hand_recognizer is not None:
            hand_recognizer.close()


if __name__ == "__main__":
    main()
