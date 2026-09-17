from pathlib import Path

from pathox.inference.slide_engine import (
    SlideInferenceEngine,
)

IMAGE_ID = "7c9d2571c6ced829b53f37465b030c5b"

SLIDE = Path(
    "/mnt/d/PathoXData/panda/radboud/images/"
    f"{IMAGE_ID}.tiff"
)

CHECKPOINT = Path(
    "/mnt/d/PathoXData/experiments/panda/"
    "pathox_panda_best.pt"
)

OUT = Path(
    "outputs/panda_slide_inference"
) / IMAGE_ID

engine = SlideInferenceEngine(
    checkpoint_path=CHECKPOINT,
    tile_size=512,
    stride=512,
    thumbnail_size=2048,
    tissue_threshold=0.12,
    batch_size=4,
    uncertainty_top_k=50,
)

result = engine.infer(
    SLIDE,
    OUT,
    image_id=IMAGE_ID,
)

print("\n" + "=" * 60)
print("PATHOX SLIDE-LEVEL INFERENCE")
print("=" * 60)

print("Slide :", result.image_id)
print("WSI   :", result.slide_dimensions)

print("\nTile analysis")
print("Candidate tiles :", result.candidate_tiles)
print("Analyzed tiles  :", result.analyzed_tiles)

print("\nQuantification")
print(
    f"Tissue fraction  : "
    f"{result.tissue_fraction:.4f}"
)
print(
    f"Predicted lesion : "
    f"{result.lesion_fraction:.4f}"
)

print("\nModel confidence / uncertainty")
print(
    f"Confidence       : "
    f"{result.mean_confidence:.4f}"
)
print(
    f"Entropy          : "
    f"{result.mean_entropy:.4f}"
)
print(
    f"Disagreement     : "
    f"{result.mean_disagreement:.4f}"
)

print("\nClass fractions")
for cls, fraction in result.class_fractions.items():
    print(
        f"  class {cls}: {fraction:.4f}"
    )

print("\nOutput:")
print(OUT)

print("\nPATHOX SLIDE INFERENCE: PASSED")
