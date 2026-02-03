import os
import cv2
import numpy as np
import mediapipe as mp
import random

# Inicializace MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints_from_video(video_path, target_len=300, visualize=False, flip_horizontal=False, augment=False):
    """
    Extrahuje body a vrací: (sekvence_bodů, délka_videa_v_sekundách).

    augment=True:
    1. Přidá náhodný šum do souřadnic.
    2. Náhodně ořízne začátek/konec videa (simulace jiné rychlosti).
    """
    cap = cv2.VideoCapture(video_path)

    # ⏱️ 1. Získání reálného času
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_orig = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count_orig / fps if fps > 0 else 0.0

    seq = []
    # Bereme jen horní polovinu těla (0-24), nohy (25+) zahazujeme
    UPPER_BODY_INDICES = list(range(0, 25))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

                # --- 📐 SHOULDER-CENTRIC NORMALIZACE ---
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                shoulder_center = (left_shoulder + right_shoulder) / 2
                landmarks -= shoulder_center  # Centrování na [0,0]

                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                scale = shoulder_width * 2.0 if shoulder_width > 1e-6 else 1.0
                landmarks /= scale  # Škálování podle velikosti postavy

                # Vybereme jen horní polovinu těla a zploštíme
                pts = landmarks[UPPER_BODY_INDICES].flatten()

                # 🎲 AUGMENTACE A: GEOMETRICKÝ ŠUM (Jitter)
                if augment:
                    noise = np.random.normal(0, 0.003, pts.shape)
                    pts += noise

                seq.append(pts)

                # Vizualizace (volitelné)
                if visualize:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.imshow("Pose detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    if visualize: cv2.destroyAllWindows()

    # Ošetření prázdného videa
    if len(seq) == 0:
        return np.zeros((target_len, 75)), 0.0

    seq = np.array(seq)

    # 🎲 AUGMENTACE B: ČASOVÉ VZORKOVÁNÍ (Time Shift)
    start_idx = 0
    end_idx = len(seq) - 1

    if augment and len(seq) > 20:
        trim = int(len(seq) * 0.15)
        start_idx = random.randint(0, trim)
        end_idx = random.randint(len(seq) - 1 - trim, len(seq) - 1)

    # Resampling (Interpolace na fixních 300 kroků)
    resampled_seq = []
    for i in range(seq.shape[1]):
        resampled_seq.append(np.interp(
            np.linspace(start_idx, end_idx, target_len),
            np.arange(len(seq)),
            seq[:, i]
        ))

    # 🎲 AUGMENTACE C: ŠUM V ČASE
    aug_duration = duration
    if augment:
        aug_duration += random.uniform(-0.5, 0.5)
        aug_duration = max(0.5, aug_duration)

    return np.array(resampled_seq).T, aug_duration


def load_dataset_multiclass(data_dir="data", target_len=300):
    """
    Původní funkce pro načítání (pro zpětnou kompatibilitu).
    Ignoruje časový údaj a vrací jen sekvence bodů.
    """
    X, y_classes = [], []

    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    if not class_names:
        print(f"❌ Žádné složky kategorií nenalezeny v '{data_dir}'!")
        return np.array([]), np.array([]), []

    for ci, cname in enumerate(class_names):
        cpath = os.path.join(data_dir, cname)
        path_left = os.path.join(cpath, "left")
        path_right = os.path.join(cpath, "right")

        def process_subfolder(subfolder_path, should_flip):
            if not os.path.exists(subfolder_path): return

            files = [f for f in os.listdir(subfolder_path) if f.endswith(".mp4") or f.endswith(".MOV")]
            print(f"📂 Zpracovávám '{cname}/{os.path.basename(subfolder_path)}': {len(files)} videí...")

            for f in files:
                full_path = os.path.join(subfolder_path, f)

                # 🛠 UPRAVENO: Funkce teď vrací (seq, duration), my chceme jen seq
                seq, _ = extract_keypoints_from_video(
                    full_path,
                    target_len=target_len,
                    visualize=False,
                    flip_horizontal=should_flip,
                    augment=False  # Pro klasický load bez augmentace
                )

                if seq.shape == (target_len, 75):
                    X.append(seq)
                    y_classes.append(ci)
                else:
                    print(f"❌ Chybný tvar dat u {f}, přeskakuji.")

        process_subfolder(path_right, should_flip=False)
        process_subfolder(path_left, should_flip=True)

    return np.stack(X), np.array(y_classes), class_names