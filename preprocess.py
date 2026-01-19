import os
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_video(video_path, target_len=300, visualize=False):
    """Extrahuje 33 klíčových bodů těla z videa (MediaPipe Pose).
       Pokud visualize=True, ukáže okno s vykreslenými body.
    """
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    seq = []
    frame_count = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            detected_frames += 1
            pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            pts = pts - np.mean(pts)  # normalizace na pohyb
            pts = pts / np.std(pts)

            seq.append(pts)

            # 🟢 Pokud chceš vizuálně vidět detekci
            if visualize:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("Pose detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC pro ukončení
                    break

    cap.release()
    cv2.destroyAllWindows()

    print(f"🎥 {video_path}: {detected_frames}/{frame_count} framů s detekovanou pózou.")

    if len(seq) == 0:
        print(f"⚠️  Žádné body detekovány ve videu: {video_path}")
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
