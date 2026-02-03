import numpy as np
from tensorflow.keras.models import load_model
from preprocess import extract_keypoints_from_video

# Načtení modelu a názvů kategorií
model = load_model("binary_classifier.keras")
class_names = np.load("class_names.npy", allow_pickle=True)

# Cesta k testovacímu videu
video_path = "test/test3.mp4"
seq = extract_keypoints_from_video(video_path, target_len=300)
seq = np.expand_dims(seq, axis=0)

# Predikce
pred = model.predict(seq)
pred = pred[0]  # vektor pravděpodobností

# Index a název nejpravděpodobnější třídy
pred_class_idx = np.argmax(pred)
pred_class_name = class_names[pred_class_idx]

# Pravděpodobnost správného postupu (pokud existuje třída 'correct')
if "correct" in class_names:
    correct_idx = class_names.tolist().index("correct")
    correct_prob = pred[correct_idx] * 100
else:
    correct_prob = pred[pred_class_idx] * 100  # fallback

# --- 💡 Výstup ---
print(f"\n🎥 Video: {video_path}")
print(f"🩺 Nejpravděpodobnější kategorie: {pred_class_name}")
print(f"✅ Shoda se správným postupem: {correct_prob:.1f}%\n")

print("📊 Podobnostní profil kategorií:")
print("-" * 45)
for name, val in zip(class_names, pred):
    print(f"{name:25s} → {val*100:6.2f} %")
print("-" * 45)
