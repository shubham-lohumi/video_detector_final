# import os
# from detector import DeepfakeDetector

# REAL_DIR = "realworld/real"
# FAKE_DIR = "realworld/fake"
# EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# def list_images(folder):
#     return [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.lower().endswith(EXTS)]

# def eval_folder(detector, folder, expected_label):
#     results = []
#     print(f"\n=== {expected_label} images ({folder}) ===")
#     for path in list_images(folder):
#         pred = detector.predict(path)
#         ok = pred["Label"] == expected_label
#         print(f"{os.path.basename(path):<22}  pred={pred['Label']:<4}  real={pred['Real Confidence']:.4f}  fake={pred['Fake Confidence']:.4f}  [{'OK' if ok else 'WRONG'}]")
#         results.append(ok)
#     return results

# if __name__ == "__main__":
#     detector = DeepfakeDetector(model_path="models/fine_tuned.pt")
#     real_hits = eval_folder(detector, REAL_DIR, "REAL")
#     fake_hits = eval_folder(detector, FAKE_DIR, "FAKE")
#     total = len(real_hits) + len(fake_hits)
#     print("\n=== Summary ===")
#     print(f"REAL accuracy: {sum(real_hits)}/{len(real_hits)} ({sum(real_hits)/len(real_hits)*100:.1f}%)")
#     print(f"FAKE accuracy: {sum(fake_hits)}/{len(fake_hits)} ({sum(fake_hits)/len(fake_hits)*100:.1f}%)")
#     print(f"Overall:       {sum(real_hits)+sum(fake_hits)}/{total} ({(sum(real_hits)+sum(fake_hits))/total*100:.1f}%)")

from detector import DeepfakeDetector

# 1. Detector initialize karein (Yeh automatically fine_tuned_v1.pt aur video_gru.pt dono utha lega)
detector = DeepfakeDetector(
    model_path="models/fine_tuned.pt",
    gru_model_path="models/video_gru.pt"
)

# 2. Apni kisi bhi test video ka path yahan dein
video_path = "realworld/fake/fake.mp4"


print(f"🔍 Analyzing video: {video_path} ...")
result = detector.predict(video_path)

print("\n--- Video Detection Result ---")
for key, value in result.items():
    print(f"{key}: {value}")