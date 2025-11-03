import numpy as np
from preprocess import load_dataset_multiclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("📦 Načítám data...")
X, y_classes, class_names = load_dataset_multiclass("data", target_len=120)
print(f"✅ Načteno {len(X)} videí, kategorie: {class_names}")

y_cat = to_categorical(y_classes, num_classes=len(class_names))
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🎯 Trénuji model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=4)

model.save("inhalation_classifier.h5")
np.save("class_names.npy", class_names)
print("💾 Model uložen jako inhalation_classifier.h5")
