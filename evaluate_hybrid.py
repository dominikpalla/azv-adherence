import numpy as np
import tensorflow as tf
from preprocess import extract_keypoints_from_video

# Načtení modelu
print("🧠 Načítám model...")
model = tf.keras.models.load_model("hybrid_classifier.keras")


def analyze_video(video_path):
    print(f"\n🎥 Analyzuji: {video_path}")

    # Extrakce (vrací body i čas)
    seq, duration = extract_keypoints_from_video(video_path, target_len=300, flip_horizontal=False)

    if seq.shape != (300, 75):
        print("⚠️ Chyba detekce postavy.")
        return

    # Příprava vstupů pro model
    # 1. Pohyb (batch dimension)
    input_motion = np.expand_dims(seq, axis=0)
    # 2. Čas (normalizovaný stejně jako při tréninku - děleno 20)
    input_time = np.array([min(duration, 20.0) / 20.0])

    print(f"   ⏱️ Délka videa: {duration:.2f} s")

    # Predikce (posíláme list dvou vstupů)
    prediction = model.predict([input_motion, input_time], verbose=0)[0][0]

    # Výsledek
    # Blízko 0 = Správně, Blízko 1 = Špatně
    percentage_error = prediction * 100
    percentage_correct = (1 - prediction) * 100

    print("-" * 30)
    print(f"   📊 Skóre chyby: {percentage_error:.1f} %")

    threshold = 0.5  # Hranice rozhodnutí

    if prediction > threshold:
        print(f"❌ VÝSLEDEK: ŠPATNĚ")
    else:
        print(f"✅ VÝSLEDEK: SPRÁVNĚ")
    print("-" * 30)


# TESTOVÁNÍ
# Změň na cesty k tvým videím
analyze_video("test/test1.mp4")
analyze_video("test/test2.mp4")
analyze_video("test/test3.mp4")