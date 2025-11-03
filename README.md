# 🫁 Inhalation AI Evaluator

Projekt pro vyhodnocování správnosti inhalace pomocí neuronové sítě a analýzy pohybu těla.

## 🔧 Instalace
```bash
pip install -r requirements.txt
```

## 🎬 Struktura dat
```
data/
├── correct/
├── no_shake/
├── too_fast_inhale/
└── wrong_mouth_position/
```
Každá složka obsahuje videa daného typu postupu.

## 🧠 Trénink
```bash
python train_full.py
```

## 🔍 Vyhodnocení nového videa
```bash
python evaluate_video.py
```

Výstup:
```
🎥 Video: test_student.mp4
✅ Shoda se správným postupem: 83.5%
🩺 Detekovaná chyba: too_fast_inhale
```

## 💡 Poznámky
- Vstupní videa by měla mít jednoho člověka v záběru (horní polovina těla).
- Oblečení nehraje roli — model používá pouze klíčové body těla.
- Můžeš přidávat libovolné nové kategorie chyb (stačí nová složka ve `data/`).
