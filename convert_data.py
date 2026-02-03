import os
import cv2


def convert_mov_to_mp4_cv2(root_folder):
    """
    Konvertuje .MOV na .mp4 pomocí OpenCV.
    Ignoruje audio stopy (řeší problém s kodekem 'apac') a maže originály.
    """
    print(f"🚀 Začínám konverzi pomocí OpenCV ve složce: {root_folder}")
    converted_count = 0
    errors = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".mov"):
                mov_path = os.path.join(root, file)
                mp4_path = os.path.splitext(mov_path)[0] + ".mp4"

                print(f"🔄 Konvertuji: {file} ...", end="", flush=True)

                try:
                    # Otevření videa
                    cap = cv2.VideoCapture(mov_path)

                    if not cap.isOpened():
                        print(" ❌ Nelze otevřít (OpenCV error).")
                        errors.append(mov_path)
                        continue

                    # Získání parametrů videa
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Fallback pro FPS, kdyby metadata chyběla
                    if fps <= 0: fps = 30.0

                    # Nastavení výstupu (kodek 'mp4v' je široce kompatibilní)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        frame_count += 1

                    # Úklid
                    cap.release()
                    out.release()

                    # Kontrola, zda se něco zapsalo
                    if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1000 and frame_count > 0:
                        print(f" ✅ Hotovo ({frame_count} framů). Mazám originál.")
                        os.remove(mov_path)
                        converted_count += 1
                    else:
                        print(f" ⚠️ Chyba: Výsledný soubor je prázdný nebo malý ({frame_count} framů).")
                        if os.path.exists(mp4_path):
                            os.remove(mp4_path)  # Smažeme vadný výstup
                        errors.append(mov_path)

                except Exception as e:
                    print(f" ❌ Chyba skriptu: {e}")
                    errors.append(mov_path)

    print("-" * 40)
    print(f"🎉 Dokončeno! Zkonvertováno: {converted_count} videí.")
    if errors:
        print(f"⚠️  Počet chyb: {len(errors)}")


if __name__ == "__main__":
    folder_to_process = "data"
    if os.path.exists(folder_to_process):
        convert_mov_to_mp4_cv2(folder_to_process)
    else:
        print(f"❌ Složka '{folder_to_process}' neexistuje.")