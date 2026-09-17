from pathlib import Path
import gc
import pandas as pd

from pathox.datasets.panda_segmentation import (
    PANDAImageMaskPair,
    PANDATileExtractor,
)

split_file = Path("/mnt/d/PathoXData/panda/radboud_split.csv")
root = Path("/mnt/d/PathoXData/panda/radboud")
out_root = Path("/mnt/d/PathoXData/processed/panda_tiles")

df = pd.read_csv(split_file)

extractor = PANDATileExtractor(
    tile_size=512,
    stride=512,
)

all_rows = []

for i, row in df.iterrows():
    image_id = row["image_id"]
    split = row["split"]

    image_path = root / "images" / f"{image_id}.tiff"
    mask_path = root / "masks" / f"{image_id}_mask.tiff"
    out = out_root / image_id

    print(
        f"[{i+1:02d}/{len(df)}] "
        f"{image_id} | ISUP {row['isup_grade']} | {split}"
    )

    try:
        rows, manifest = extractor.extract(
            PANDAImageMaskPair(
                image_id=image_id,
                image_path=image_path,
                mask_path=mask_path,
            ),
            out,
            per_class=8,
        )

        for r in rows:
            r["split"] = split
            r["isup_grade"] = int(row["isup_grade"])
            r["gleason_score"] = row["gleason_score"]
            all_rows.append(r)

        print(f"   tiles: {len(rows)}")

    except Exception as e:
        print(f"   ERROR: {e}")

    gc.collect()

combined = pd.DataFrame(all_rows)

combined_path = out_root / "panda_segmentation_manifest.csv"
combined.to_csv(combined_path, index=False)

print("\n" + "=" * 60)
print("BATCH EXTRACTION SUMMARY")
print("=" * 60)
print("Slides processed :", len(df))
print("Total tiles      :", len(combined))

if len(combined):
    print("\nTiles by split:")
    print(combined["split"].value_counts().to_string())

    print("\nTiles by target class:")
    print(combined["target_class"].value_counts().sort_index().to_string())

    print("\nSlides represented:")
    print(combined["image_id"].nunique())

print("\nManifest:")
print(combined_path)
