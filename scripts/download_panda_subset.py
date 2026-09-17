from pathlib import Path
import subprocess
import pandas as pd
import zipfile
import shutil

SUBSET = Path("/mnt/d/PathoXData/panda/radboud_subset.csv")
ROOT = Path("/mnt/d/PathoXData/panda/radboud")
IMAGES = ROOT / "images"
MASKS = ROOT / "masks"

IMAGES.mkdir(parents=True, exist_ok=True)
MASKS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SUBSET)

valid = []
failed = []

for i, row in df.iterrows():
    image_id = row["image_id"]
    image_path = IMAGES / f"{image_id}.tiff"
    mask_path = MASKS / f"{image_id}_mask.tiff"

    print(f"[{i+1:02d}/{len(df)}] {image_id} | ISUP {row['isup_grade']}")

    # Download WSI if needed.
    if not image_path.exists():
        r = subprocess.run(
            [
                "kaggle", "competitions", "download",
                "prostate-cancer-grade-assessment",
                "-f", f"train_images/{image_id}.tiff",
                "-p", str(IMAGES),
            ],
            capture_output=True,
            text=True,
        )

        if r.returncode == 0:
            z = IMAGES / f"{image_id}.tiff.zip"
            if z.exists():
                with zipfile.ZipFile(z) as f:
                    f.extractall(IMAGES)
                z.unlink()

    # Download mask.
    if not mask_path.exists():
        r = subprocess.run(
            [
                "kaggle", "competitions", "download",
                "prostate-cancer-grade-assessment",
                "-f", f"train_label_masks/{image_id}_mask.tiff",
                "-p", str(MASKS),
            ],
            capture_output=True,
            text=True,
        )

        if r.returncode == 0:
            z = MASKS / f"{image_id}_mask.tiff.zip"
            if z.exists():
                with zipfile.ZipFile(z) as f:
                    f.extractall(MASKS)
                z.unlink()

    if image_path.exists() and mask_path.exists():
        valid.append(image_id)
        print("   OK")
    else:
        failed.append(image_id)
        print("   skipped")

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
print("Requested :", len(df))
print("Valid     :", len(valid))
print("Skipped   :", len(failed))

print("\nValid IDs:")
for x in valid:
    print(x)

pd.DataFrame({"image_id": valid}).to_csv(
    "/mnt/d/PathoXData/panda/radboud_valid_ids.csv",
    index=False,
)

print("\nSaved:")
print("/mnt/d/PathoXData/panda/radboud_valid_ids.csv")

print("\nDisk usage:")
subprocess.run(["du", "-sh", str(ROOT)])
