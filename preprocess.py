import os
import cv2
import numpy as np
import mediapipe as mp

# Inicializace MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints_from_video(video_path, target_len=300, visualize=False, flip_horizontal=False):
    """
    Extrahuje klíčové body z videa s důrazem na horní polovinu těla.

    VYLEPŠENÍ:
    1. Shoulder-Centric Normalizace: Střed souřadnic (0,0) je bod mezi rameny.
    2. Odstranění nohou: Zahazuje body 25+ (kolena, kotníky), které pod stolem dělají nepořádek.
    3. Výstup: Matice (target_len, 75) -> 25 bodů * 3 souřadnice (x, y, z).
    """
    cap = cv2.VideoCapture(video_path)
    seq = []
    frame_count = 0
    detected_frames = 0

    # Které body chceme nechat? (0 až 24 = Hlava, Ruce, Trup, Kyčle)
    # Vyhazujeme 25 až 32 (Nohy)
    UPPER_BODY_INDICES = list(range(0, 25))

    # Použití 'with' zajistí správné uvolnění zdrojů MediaPipe
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 1 je default, 2 je přesnější ale pomalejší
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 🔄 Transformace pro leváky (zrcadlení obrazu)
            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            # MediaPipe vyžaduje RGB
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                detected_frames += 1

                # 1. Získání surových dat (33 bodů x 3 souřadnice)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

                # --- 📐 SHOULDER-CENTRIC NORMALIZACE ---

                # A. Získáme ramena (body 11 a 12)
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]

                # B. Vypočítáme střed mezi rameny (naše nová nula [0,0,0])
                shoulder_center = (left_shoulder + right_shoulder) / 2

                # C. CENTROVÁNÍ: Posuneme všechny body
                landmarks -= shoulder_center

                # D. ŠKÁLOVÁNÍ: Podle šířky ramen
                # Díky tomu je jedno, jak blízko kameře člověk stojí
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

                # Ochrana proti dělení nulou (extrémní úhel z boku)
                if shoulder_width < 1e-6:
                    scale = 1.0
                else:
                    # Násobíme 2.0, aby data byla v rozsahu cca -1 až 1
                    scale = shoulder_width * 2.0

                landmarks /= scale

                # --- ✂️ OŘEZ DATASETU (Jen horní polovina) ---
                landmarks_upper = landmarks[UPPER_BODY_INDICES]

                # Zploštění do vektoru (25 bodů * 3 = 75 hodnot)
                pts = landmarks_upper.flatten()
                seq.append(pts)

                # Vizualizace pro debug (zpomaluje trénink, nechat False)
                if visualize:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.imshow("Pose detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC pro ukončení
                        break

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    # print(f"🎥 {os.path.basename(video_path)}: {detected_frames}/{frame_count} frames.")

    # ⚠️ Ošetření prázdného videa nebo videa bez detekce
    if len(seq) == 0:
        print(f"⚠️  Žádné body detekovány: {video_path}")
        return np.zeros((target_len, 75))  # Vracíme nuly správné velikosti

    # Normalizace délky sekvence (Resampling na target_len)
    # Toto zajistí, že LSTM dostane vždy 300 kroků, ať je video dlouhé jakkoliv
    seq = np.array(seq)
    resampled_seq = []
    for i in range(seq.shape[1]):  # Pro každou souřadnici (0..74)
        resampled_seq.append(np.interp(
            np.linspace(0, len(seq) - 1, target_len),
            np.arange(len(seq)),
            seq[:, i]
        ))

    # Transpozice zpět na tvar (čas, features) -> (300, 75)
    return np.array(resampled_seq).T


def load_dataset_multiclass(data_dir="data", target_len=300):
    """
    Načte videa ze struktury:
    data/
      ├── kategorie_A/
      │     ├── left/   (bude transformováno)
      │     └── right/  (původní)
      ...
    """
    X, y_classes = [], []

    # Získání seznamu kategorií (seřazeno abecedně pro konzistenci)
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    if not class_names:
        print(f"❌ Žádné složky kategorií nenalezeny v '{data_dir}'!")
        return np.array([]), np.array([]), []

    for ci, cname in enumerate(class_names):
        cpath = os.path.join(data_dir, cname)

        # Cesty k podsložkám
        path_left = os.path.join(cpath, "left")
        path_right = os.path.join(cpath, "right")

        # Pomocná vnitřní funkce
        def process_subfolder(subfolder_path, should_flip):
            if not os.path.exists(subfolder_path):
                return

            files = [f for f in os.listdir(subfolder_path) if f.endswith(".mp4") or f.endswith(".MOV")]
            print(f"📂 Zpracovávám '{cname}/{os.path.basename(subfolder_path)}': {len(files)} videí...")

            for f in files:
                full_path = os.path.join(subfolder_path, f)

                # 🚀 Hlavní volání extrakce
                seq = extract_keypoints_from_video(
                    full_path,
                    target_len=target_len,
                    visualize=False,
                    flip_horizontal=should_flip
                )

                # Kontrola integrity dat
                if seq.shape == (target_len, 75):
                    X.append(seq)
                    y_classes.append(ci)
                else:
                    print(f"❌ Chybný tvar dat u {f}, přeskakuji.")

        # 1. Zpracuj praváky (bez flipu)
        process_subfolder(path_right, should_flip=False)

        # 2. Zpracuj leváky (s flipem)
        process_subfolder(path_left, should_flip=True)

    if len(X) == 0:
        print("❌ Nebyla načtena žádná data! Zkontroluj strukturu složek.")
        return np.array([]), np.array([]), class_names

    return np.stack(X), np.array(y_classes), class_names