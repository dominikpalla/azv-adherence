from preprocess import extract_keypoints_from_video
import glob

# Najde všechna videa ve složce data
videos = glob.glob("test/spatne2.mp4", recursive=True)

print(f"\n🔍 Nalezeno {len(videos)} videí ke kontrole:\n")
for v in videos:
    extract_keypoints_from_video(v, visualize=True)
