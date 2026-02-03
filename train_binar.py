# -------------------------------------------------
# ☯️ TRAIN_BINARY.PY – Správně (0) vs. Špatně (1)
# -------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess
import unicodedata
from preprocess import extract_keypoints_from_video  # Používá tvůj aktuální preprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# -------------------------------------------------
# ☕️ ANTI-SLEEP (M4 Pro)
# -------------------------------------------------
print("🚀 Aktivuji CAFFEINATE...")
caffeinate_process = subprocess.Popen(['caffeinate', '-d'])

try:
    # Fixní seed
    np.random.seed(42)
    tf.random.set_seed(42)


    # -------------------------------------------------
    # 🔄 Funkce pro binární načítání
    # -------------------------------------------------
    def load_binary_dataset(data_dir="data", target_len=300):
        X = []
        y = []  # 0 = Správně, 1 = Špatně

        # Získáme všechny složky
        folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        print(f"📂 Nalezeno {len(folders)} kategorií. Převádím na BINÁRNÍ...")

        for folder in folders:
            # 🔧 OPRAVA PRO MAC: Normalizace Unicode (sjednotí háčky a čárky)
            folder_normalized = unicodedata.normalize('NFC', folder).lower()

            # Určení labelu podle názvu složky
            if "správně" in folder_normalized:
                label = 0
                label_name = "✅ SPRÁVNĚ"
            else:
                label = 1
                label_name = "❌ ŠPATNĚ"

            # Cesta ke složce
            folder_path = os.path.join(data_dir, folder)

            # Projdeme podsložky left/right (pokud existují)
            subdirs = ["left", "right"]
            for subdir in subdirs:
                subpath = os.path.join(folder_path, subdir)
                if not os.path.exists(subpath): continue

                should_flip = (subdir == "left")
                files = [f for f in os.listdir(subpath) if f.endswith(".mp4") or f.endswith(".MOV")]

                print(f"   -> {label_name} ({folder}/{subdir}): {len(files)} videí")

                for f in files:
                    full_path = os.path.join(subpath, f)
                    seq = extract_keypoints_from_video(full_path, target_len, flip_horizontal=should_flip)

                    if seq.shape == (target_len, 75):
                        X.append(seq)
                        y.append(label)

        return np.array(X), np.array(y)


    # -------------------------------------------------
    # 📦 Načtení dat
    # -------------------------------------------------
    print("📦 Načítám data...")
    X, y = load_binary_dataset("data", target_len=300)

    print(f"\n📊 Celkem videí: {len(X)}")
    count_ok = np.sum(y == 0)
    count_ko = np.sum(y == 1)
    print(f"✅ Správně: {count_ok}")
    print(f"❌ Špatně:  {count_ko}")

    if count_ok == 0 or count_ko == 0:
        raise ValueError("Chybí data pro jednu z tříd! Zkontroluj názvy složek.")

    # Rozdělení na trénink a test
    # Stratify zajistí, že v testu bude stejný poměr správně/špatně jako v tréninku
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y
    )

    # -------------------------------------------------
    # ⚖️ Výpočet vah (Balancing Class Imbalance)
    # -------------------------------------------------
    # Toto zajistí, že model nebude ignorovat menšinovou třídu (Správně)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"⚖️ Váhy tříd: {class_weights_dict}")
    # (Pravděpodobně to vyjde něco jako {0: 3.5, 1: 0.6})

    # -------------------------------------------------
    # 🏗️ Definice modelu (Binární verze)
    # -------------------------------------------------
    model = Sequential([
        Input(shape=(300, 75)),  # 300 framů, 75 souřadnic
        GaussianNoise(0.05),  # Šum proti přeučení

        LSTM(64, return_sequences=True),
        Dropout(0.4),

        LSTM(32),
        Dropout(0.4),

        # BINÁRNÍ VÝSTUP: 1 neuron, sigmoid (0 až 1)
        Dense(1, activation='sigmoid')
    ])

    # Používáme Binary Crossentropy
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    # -------------------------------------------------
    # 🚀 Trénink
    # -------------------------------------------------
    print("🎯 Trénuji BINÁRNÍ model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=8,
        class_weight=class_weights_dict,  # <--- ZDE APLIKUJEME VÁHY
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # -------------------------------------------------
    # 💾 Uložení a Grafy
    # -------------------------------------------------
    model.save("binary_classifier.keras")
    print("💾 Model uložen jako binary_classifier.keras")

    plt.figure(figsize=(12, 5))

    # Graf Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy (Binární)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graf Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('binary_training_plot.png', dpi=150)
    print("📊 Graf uložen jako binary_training_plot.png")

except Exception as e:
    print(f"❌ CHYBA: {e}")

finally:
    caffeinate_process.terminate()