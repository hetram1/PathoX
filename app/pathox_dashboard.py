from pathlib import Path
import json
import pandas as pd

import streamlit as st

from pathox.inference.slide_engine import SlideInferenceEngine


DEFAULT_SLIDE = (
    "/mnt/d/PathoXData/panda/radboud/images/"
    "7c9d2571c6ced829b53f37465b030c5b.tiff"
)

DEFAULT_CHECKPOINT = (
    "/mnt/d/PathoXData/experiments/panda/"
    "pathox_panda_best.pt"
)


st.set_page_config(
    page_title="PathoX",
    page_icon="🧬",
    layout="wide",
)

st.title("PathoX — Gigapixel Digital Pathology AI Engine")
st.caption(
    "Whole-Slide Image segmentation, uncertainty analysis, "
    "and pathology quantification"
)

with st.sidebar:
    st.header("Inference Configuration")

    slide_path = st.text_input(
        "WSI path",
        value=DEFAULT_SLIDE,
    )

    checkpoint_path = st.text_input(
        "Model checkpoint",
        value=DEFAULT_CHECKPOINT,
    )

    tile_size = st.number_input(
        "Tile size",
        min_value=128,
        max_value=1024,
        value=512,
        step=128,
    )

    tissue_threshold = st.slider(
        "Tissue threshold",
        min_value=0.01,
        max_value=0.50,
        value=0.12,
        step=0.01,
    )

    uncertainty_top_k = st.number_input(
        "Uncertainty tiles",
        min_value=5,
        max_value=200,
        value=50,
        step=5,
    )

    run = st.button(
        "Run PathoX Inference",
        type="primary",
        width='stretch',
    )


if run:
    slide = Path(slide_path)
    checkpoint = Path(checkpoint_path)

    if not slide.exists():
        st.error(f"WSI not found: {slide}")
        st.stop()

    if not checkpoint.exists():
        st.error(
            f"Checkpoint not found: {checkpoint}"
        )
        st.stop()

    image_id = slide.stem

    output_dir = (
        Path("outputs/dashboard")
        / image_id
    )

    with st.spinner(
        "Running tissue screening, segmentation, "
        "uncertainty analysis and quantification..."
    ):
        try:
            engine = SlideInferenceEngine(
                checkpoint_path=checkpoint,
                tile_size=int(tile_size),
                stride=int(tile_size),
                thumbnail_size=2048,
                tissue_threshold=float(
                    tissue_threshold
                ),
                batch_size=4,
                uncertainty_top_k=int(
                    uncertainty_top_k
                ),
            )

            result = engine.infer(
                slide,
                output_dir,
                image_id=image_id,
            )

        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.success("PathoX inference completed.")

    st.subheader("Slide Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "WSI size",
        f"{result.slide_dimensions[0]} × "
        f"{result.slide_dimensions[1]}",
    )

    c2.metric(
        "Candidate tiles",
        result.candidate_tiles,
    )

    c3.metric(
        "Predicted non-background fraction",
        f"{result.lesion_fraction:.2%}",
    )

    c4.metric(
        "Confidence",
        f"{result.mean_confidence:.2%}",
    )

    st.subheader("Model Uncertainty")

    u1, u2 = st.columns(2)

    u1.metric(
        "Mean entropy",
        f"{result.mean_entropy:.4f}",
    )

    u2.metric(
        "TTA disagreement",
        f"{result.mean_disagreement:.4f}",
    )

    st.subheader("Class Distribution")

    class_rows = [
        {
            "Class": str(cls),
            "Fraction": fraction,
        }
        for cls, fraction
        in result.class_fractions.items()
    ]

    st.dataframe(
        class_rows,
        width='stretch',
        hide_index=True,
    )

    overview = (
        output_dir / "slide_overview.png"
    )
    lesion_heatmap = (
        output_dir / "lesion_heatmap.png"
    )
    uncertainty_heatmap = (
        output_dir / "uncertainty_heatmap.png"
    )

    if overview.exists():
        st.subheader("WSI Overview")
        st.image(
            str(overview),
            width='stretch',
        )

    st.subheader("PathoX Heatmaps")

    h1, h2 = st.columns(2)

    if lesion_heatmap.exists():
        h1.image(
            str(lesion_heatmap),
            caption="Predicted non-background fraction",
            width='stretch',
        )

    if uncertainty_heatmap.exists():
        h2.image(
            str(uncertainty_heatmap),
            caption="Model uncertainty",
            width='stretch',
        )

    hard_csv = (
        output_dir / "hard_tiles.csv"
    )

    if hard_csv.exists():
        st.subheader(
            "Highest-Uncertainty Regions"
        )

        st.dataframe(
            pd.read_csv(hard_csv),
            width="stretch",
            hide_index=True,
        )

    report = (
        output_dir / "slide_report.json"
    )

    if report.exists():
        with st.expander(
            "Raw PathoX report"
        ):
            st.json(
                json.loads(
                    report.read_text()
                )
            )

else:
    st.info(
        "Enter a WSI path in the sidebar "
        "and click 'Run PathoX Inference'."
    )

    st.markdown(
        """
### PathoX Pipeline

**WSI → Tissue Screening → Tile Inference → "
Segmentation → Uncertainty → Quantification → Heatmaps**
"""
    )
