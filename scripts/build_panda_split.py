from pathlib import Path
import pandas as pd

src = Path("/mnt/d/PathoXData/panda/radboud_subset.csv")
out = Path("/mnt/d/PathoXData/panda/radboud_split.csv")

df = pd.read_csv(src)

# 1 validation slide + 3 training slides per ISUP grade.
val = (
    df.groupby("isup_grade", group_keys=False)
      .sample(n=1, random_state=123)
      .copy()
)

df["split"] = "train"
df.loc[df["image_id"].isin(val["image_id"]), "split"] = "val"

df = df.sort_values(["split", "isup_grade", "image_id"])
df.to_csv(out, index=False)

print("Total slides :", len(df))
print("\nSplit counts:")
print(df["split"].value_counts().to_string())

print("\nSplit × ISUP:")
print(pd.crosstab(df["split"], df["isup_grade"]).to_string())

print("\nValidation slides:")
print(
    df.loc[df["split"] == "val",
           ["image_id", "isup_grade", "gleason_score"]]
      .to_string(index=False)
)

print("\nSaved:", out)
