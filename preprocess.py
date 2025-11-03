import os
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_keypoints_from_video(video_path, target_len=120):
    """Extrahuje 33 klíčových bodů těla z videa (MediaPipe Pose)."""
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            seq.append(pts)
    cap.release()

    if len(seq) == 0:
        print(f"[WARN] Ve videu {video_path} nebyly nalezeny body.")
        return np.zeros((target_len, 99))

    # Normalizace délky sekvence
    idx = np.linspace(0, len(seq)-1, target_len).astype(int)
    return np.array(seq)[idx]


def load_dataset_multiclass(data_dir="data", target_len=120):
    """
    Načte všechna videa podle složek (každá složka = kategorie).
    Vrací X (sekvence) a y (třídy).
    """
    X, y_classes = [], []
    class_names = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    for ci, cname in enumerate(class_names):
        cpath = os.path.join(data_dir, cname)
        for f in os.listdir(cpath):
            if not f.endswith(".mp4"):
                continue
            seq = extract_keypoints_from_video(os.path.join(cpath, f), target_len)
            X.append(seq)
            y_classes.append(ci)
    return np.stack(X), np.array(y_classes), class_names
