# -------------------------------------------------
# 🧠 TRAIN_FULL.PY – Stabilní trénink modelu pohybových sekvencí
# -------------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_dataset_multiclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# ⚙️ Fixní semínka pro reprodukovatelnost
# -------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------
# 📦 Načtení dat
# -------------------------------------------------
print("📦 Načítám data...")
X, y_classes, class_names = load_dataset_multiclass("data", target_len=300)
print(f"✅ Načteno {len(X)} videí, kategorie: {class_names}")

# Převod cílových tříd na one-hot vektory
y_cat = to_categorical(y_classes, num_classes=len(class_names))

# Rozdělení dat na trénink a test (30 % test = validace)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42, shuffle=True)
print(f"📊 Rozdělení: {len(X_train)} tréninkových, {len(X_test)} validačních vzorků")

# -------------------------------------------------
# 🏗️ Definice modelu
# -------------------------------------------------
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------------------------
# 🧩 Early Stopping – zastaví, když se validace 10 epoch nezlepšuje
# -------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# -------------------------------------------------
# 🚀 Trénink
# -------------------------------------------------
print("🎯 Trénuji model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=4,
    #callbacks=[early_stop],
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
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val acc', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val loss', linewidth=2)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()
print("📊 Graf průběhu uložen jako training_plot.png")

# -------------------------------------------------
# 🧩 Poznámka: Pokud MediaPipe hází chybu 'GetPrototype',
# spusť jednou v terminálu:
# pip install --upgrade protobuf==3.20.3 mediapipe
# -------------------------------------------------
