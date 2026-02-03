import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess
import unicodedata
from preprocess import extract_keypoints_from_video
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GaussianNoise, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

print("🚀 Aktivuji CAFFEINATE...")
caffeinate_process = subprocess.Popen(['caffeinate', '-d'])

try:
    np.random.seed(42)
    tf.random.set_seed(42)


    # -------------------------------------------------
    # 🔄 Načítání s Augmentací
    # -------------------------------------------------
    def load_augmented_dataset(data_dir="data", target_len=300):
        X_motion = []
        X_time = []
        y = []

        folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"📂 Zpracovávám složky...")

        for folder in folders:
            # Normalizace názvu (kvůli Macu)
            folder_norm = unicodedata.normalize('NFC', folder).lower()

            # Je to "Správně"? Pokud ano, budeme augmentovat (násobit data)
            is_correct = "správně" in folder_norm
            label = 0 if is_correct else 1

            # Kolikrát video použít?
            # Správně -> 5x (1x originál + 4x augmentace) -> Vyrovná poměr 100:500 na 500:500
            # Špatně  -> 1x (jen originál)
            repeats = 5 if is_correct else 1

            folder_path = os.path.join(data_dir, folder)

            for subdir in ["left", "right"]:
                subpath = os.path.join(folder_path, subdir)
                if not os.path.exists(subpath): continue

                # Vše normalizujeme na praváky
                should_flip = (subdir == "left")
                files = [f for f in os.listdir(subpath) if f.endswith(".mp4") or f.endswith(".MOV")]

                print(f"   -> {folder}/{subdir}: {len(files)} videí (Generuji {repeats}x)")

                for f in files:
                    full_path = os.path.join(subpath, f)

                    # 1. Průchod: Originál (vždy)
                    seq, dur = extract_keypoints_from_video(full_path, target_len, flip_horizontal=should_flip,
                                                            augment=False)
                    if seq.shape == (target_len, 75):
                        X_motion.append(seq)
                        X_time.append(min(dur, 20.0) / 20.0)  # Normalizace času (0-1)
                        y.append(label)

                    # 2. Průchod: Augmentované kopie (jen pro Správně)
                    if repeats > 1:
                        for _ in range(repeats - 1):
                            seq_aug, dur_aug = extract_keypoints_from_video(full_path, target_len,
                                                                            flip_horizontal=should_flip, augment=True)
                            if seq_aug.shape == (target_len, 75):
                                X_motion.append(seq_aug)
                                X_time.append(min(dur_aug, 20.0) / 20.0)
                                y.append(label)

        return np.array(X_motion), np.array(X_time), np.array(y)


    # -------------------------------------------------
    # 📦 Příprava dat
    # -------------------------------------------------
    print("📦 Načítám a generuji data...")
    X_motion, X_time, y = load_augmented_dataset("data")

    print(f"\n📊 Celkem vzorků: {len(y)}")
    print(f"✅ Správně (vč. augmentace): {np.sum(y == 0)}")
    print(f"❌ Špatně: {np.sum(y == 1)}")

    # Split
    # Musíme rozdělit oba vstupy (Motion i Time)
    Xm_train, Xm_test, Xt_train, Xt_test, y_train, y_test = train_test_split(
        X_motion, X_time, y, test_size=0.25, random_state=42, shuffle=True, stratify=y
    )

    # Váhy (i když jsme data namnožili, necháme jemné váhy pro jistotu)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = dict(enumerate(class_weights))
    print(f"⚖️ Váhy: {weights_dict}")

    # -------------------------------------------------
    # 🏗️ Hybridní Model (Pohyb + Čas)
    # -------------------------------------------------
    # Vstup 1: Pohyb
    input_motion = Input(shape=(300, 75), name="motion_input")
    x = GaussianNoise(0.05)(input_motion)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.4)(x)
    x = LSTM(32)(x)
    x = Dropout(0.4)(x)

    # Vstup 2: Čas
    input_time = Input(shape=(1,), name="time_input")
    t = Dense(8, activation="relu")(input_time)

    # Spojení
    combined = Concatenate()([x, t])
    z = Dense(32, activation="relu")(combined)
    output = Dense(1, activation="sigmoid")(z)  # 0=Správně, 1=Špatně

    model = Model(inputs=[input_motion, input_time], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # -------------------------------------------------
    # 🚀 Trénink
    # -------------------------------------------------
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    print("🎯 Trénuji HYBRIDNÍ model...")
    history = model.fit(
        [Xm_train, Xt_train], y_train,  # Dva vstupy!
        validation_data=([Xm_test, Xt_test], y_test),
        epochs=80,
        batch_size=8,
        class_weight=weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    model.save("hybrid_classifier.keras")
    print("💾 Uloženo: hybrid_classifier.keras")

    # Grafy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.savefig('hybrid_training_plot.png')

except Exception as e:
    print(f"❌ CHYBA: {e}")
finally:
    caffeinate_process.terminate()