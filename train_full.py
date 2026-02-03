# -------------------------------------------------
# 🧠 TRAIN_FULL.PY – Vylepšená verze (Anti-Sleep + Early Stopping)
# -------------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess  # <--- NOVINKA: Pro ovládání systému
import time  # <--- NOVINKA: Pro jistotu
from preprocess import load_dataset_multiclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GaussianNoise
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# ☕️ ANTI-SLEEP MODE (Pro M4 Pro)
# -------------------------------------------------
print("🚀 Aktivuji CAFFEINATE - zabraňuji uspání MacBooku...")
# Spustí proces na pozadí, který drží Mac vzhůru (-d = display on)
caffeinate_process = subprocess.Popen(['caffeinate', '-d'])

try:
    # -------------------------------------------------
    # ⚙️ Fixní semínka pro reprodukovatelnost
    # -------------------------------------------------
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------------------------------
    # 📦 Načtení dat
    # -------------------------------------------------
    print("📦 Načítám data...")
    # Předpokládám, že data jsou (Vzorky, Čas, Rysy)
    X, y_classes, class_names = load_dataset_multiclass("data", target_len=300)
    print(f"✅ Načteno {len(X)} videí, kategorie: {class_names}")

    # Převod cílových tříd na one-hot vektory
    y_cat = to_categorical(y_classes, num_classes=len(class_names))

    # Rozdělení dat
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42, shuffle=True)
    print(f"📊 Rozdělení: {len(X_train)} tréninkových, {len(X_test)} validačních vzorků")

    # -------------------------------------------------
    # 🏗️ Definice modelu (S lehkou úpravou proti overfittingu)
    # -------------------------------------------------
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),

        # <--- NOVINKA: GaussianNoise přidá trochu šumu do trénovacích dat.
        # Pomáhá modelu nebiflovat se přesná čísla, ale hledat vzory.
        GaussianNoise(0.05),

        LSTM(128, return_sequences=True),
        Dropout(0.4),  # <--- ZVÝŠENO z 0.3 na 0.4 (agresivnější zapomínání)

        LSTM(64),
        Dropout(0.4),  # <--- ZVÝŠENO z 0.3 na 0.4

        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 🧩 Early Stopping – ODKOMENTOVÁNO
    # -------------------------------------------------
    # restore_best_weights=True je KLÍČOVÉ. I když to stopne v 50. epoše,
    # vrátí ti to váhy z té nejlepší epochy (třeba 35.), ne ty poslední zhoršené.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Dáme mu trochu víc času na vzpamatování
        restore_best_weights=True,
        verbose=1
    )

    # -------------------------------------------------
    # 🚀 Trénink
    # -------------------------------------------------
    print("🎯 Trénuji model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # Zvýšeno, protože EarlyStopping to stejně utne
        batch_size=8,  # M4 Pro zvládne víc než 4, zrychlí to trénink
        callbacks=[early_stop],  # <--- AKTIVOVÁNO
        verbose=1
    )

    # -------------------------------------------------
    # 💾 Uložení modelu
    # -------------------------------------------------
    model.save("inhalation_classifier.keras")
    np.save("class_names.npy", class_names)
    print("💾 Model uložen jako inhalation_classifier.keras")

    # -------------------------------------------------
    # 📈 Graf průběhu tréninku
    # -------------------------------------------------
    plt.figure(figsize=(12, 5))  # Trochu širší graf

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val acc', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=150)
    # plt.show() # Na serveru/pozadí raději neblokovat
    print("📊 Graf průběhu uložen jako training_plot.png")

except Exception as e:
    print(f"❌ CHYBA: {e}")

finally:
    # -------------------------------------------------
    # 🛌 Vypnutí nespavosti (i když to spadne)
    # -------------------------------------------------
    print("✅ Hotovo (nebo pád). Vypínám Caffeinate, Mac může spát.")
    caffeinate_process.terminate()