# PathoX — Gigapixel Digital Pathology AI Engine

PathoX is an end-to-end digital pathology AI engine for processing large
whole-slide images (WSIs) using tissue-aware tiling, GPU segmentation,
uncertainty estimation, whole-slide quantification, and native C++/OpenMP
acceleration.

The system is designed around the practical constraints of pathology WSIs:
very large image dimensions, sparse tissue regions, expensive tile processing,
and the need to convert tile-level predictions into slide-level measurements.

---

## Overview

PathoX combines:

- Pyramid-aware whole-slide image processing
- Tissue-aware tile screening
- Native C++17 / OpenMP acceleration
- GPU-based semantic segmentation
- Test-time augmentation and uncertainty estimation
- Whole-slide quantitative aggregation
- Heatmap and JSON report generation
- Streamlit visualization
- Docker deployment with NVIDIA GPU support

---

## Pipeline

~~~text
                Whole-Slide Image
                        │
                        ▼
             Pyramid-aware WSI Reader
                        │
                        ▼
              Thumbnail Tissue Mask
                        │
                        ▼
          Tissue-aware Tile Candidate Scan
                        │
                 C++ / OpenMP
                        │
                        ▼
               512×512 Image Tiles
                        │
                        ▼
             GPU Segmentation Model
                Compact U-Net
                        │
                        ▼
             Tile-level Probabilities
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Quantification       Uncertainty
                              TTA / Entropy
              │                   │
              └─────────┬─────────┘
                        ▼
              Whole-slide Aggregation
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       CSV Data      Heatmaps      JSON Report
                        │
                        ▼
                  Streamlit UI
~~~

---

## Key Features

### Whole-slide image processing

- Pyramid-aware WSI reading with OpenSlide
- Region extraction without loading an entire gigapixel slide into RAM
- Configurable tile size and stride
- Tissue-aware candidate selection
- Slide-level prediction aggregation

### Tissue processing

- RGB/HSV-based tissue detection
- Morphological cleanup
- Connected-component filtering
- Tissue-fraction scoring
- Low-resolution thumbnail screening before expensive inference

### AI segmentation

- PyTorch compact U-Net
- Six-class prostate tissue segmentation for the PANDA Radboud subset
- Combined Dice + Cross-Entropy segmentation loss
- Class-aware tile sampling
- CUDA GPU inference
- Independent validation evaluation

### Uncertainty estimation

- Test-time augmentation (TTA)
- Predictive entropy
- Mean confidence
- TTA disagreement
- Hard-tile selection for inspection

### Native acceleration

- C++17 extension through pybind11
- OpenMP parallel tile scoring
- Native tissue-aware candidate selection
- Python/C++ correctness checks
- Native vs Python performance benchmarks

### Deployment

- Dockerized application
- NVIDIA GPU access through Docker Compose
- Streamlit dashboard
- External pathology data mounted read-only
- Output artifacts written to the host

---

## Technology Stack

| Area | Technology |
|---|---|
| Languages | Python, C++ |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, scikit-image |
| Whole-Slide Imaging | OpenSlide |
| Native Acceleration | C++17, OpenMP, pybind11 |
| Numerical/Data | NumPy, Pandas, SciPy |
| Evaluation | scikit-learn |
| UI | Streamlit |
| Build | CMake |
| Packaging | setuptools |
| Deployment | Docker, Docker Compose |
| Hardware | CUDA GPU |

---

## Architecture

~~~text
┌──────────────────────────────────────────────────────────────┐
│                    Whole-Slide Image                        │
│                     OpenSlide Reader                        │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 Thumbnail Tissue Detection                  │
│            RGB / HSV + morphology + components              │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│             Tissue-aware Candidate Tile Selection           │
│                   C++17 + OpenMP                            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Tile Extraction                         │
│                    512 × 512 patches                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    GPU Segmentation                         │
│                     Compact U-Net                           │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Prediction + TTA                            │
│       Confidence / Entropy / TTA Disagreement               │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 Whole-Slide Aggregation                     │
│       Class fractions / heatmaps / reports / hard tiles     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
└──────────────────────────────────────────────────────────────┘
~~~

---

## Project Structure

~~~text
PathoX/
├── app/
│   └── pathox_dashboard.py
│
├── cpp/
│   └── pathox_native/
│       ├── CMakeLists.txt
│       └── bindings.cpp
│
├── src/
│   └── pathox/
│       ├── config.py
│       │
│       ├── core/
│       │   ├── wsi.py
│       │   ├── tile.py
│       │   ├── tissue.py
│       │   ├── tissue_filter.py
│       │   └── stain.py
│       │
│       ├── datasets/
│       │   ├── panda.py
│       │   ├── split.py
│       │   ├── panda_segmentation.py
│       │   └── segmentation_dataset.py
│       │
│       ├── inference/
│       │   ├── slide_engine.py
│       │   └── uncertainty.py
│       │
│       ├── models/
│       │   ├── unet.py
│       │   └── losses.py
│       │
│       ├── native/
│       │   └── __init__.py
│       │
│       └── training/
│           └── engine.py
│
├── scripts/
│   ├── train_real_panda.py
│   ├── train_real_panda_weighted.py
│   ├── evaluate_panda_model.py
│   ├── run_slide_inference.py
│   ├── uncertainty_hard_tiles.py
│   ├── benchmark_cpp_acceleration.py
│   ├── benchmark_slide_candidate_selection.py
│   └── benchmark_end_to_end.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
~~~

---

## Dataset

PathoX was validated using a small real-data subset of the
PANDA prostate cancer dataset.

Current experiment:

- 24 Radboud slides downloaded
- 4 slides selected for each ISUP grade
- 18 slides used for training
- 6 slides used for validation
- 786 real 512×512 tiles extracted
- 591 training tiles
- 195 validation tiles
- Six segmentation classes

The dataset is intentionally kept outside Git because whole-slide images and
segmentation masks are large research artifacts.

Raw data, checkpoints, and generated outputs are excluded from the repository.

---

## Dataset Split

The train/validation split is performed at the slide level rather than at the
tile level to reduce leakage between correlated tiles from the same WSI.

The current split contains:

~~~text
Training slides   : 18
Validation slides : 6

Training tiles    : 591
Validation tiles  : 195
Total tiles       : 786
~~~

---

## Segmentation Model

PathoX currently uses a compact U-Net implemented in PyTorch.

The model is designed for the engineering prototype rather than maximum
model capacity, allowing the complete WSI pipeline to remain practical on a
6 GB laptop GPU.

The current segmentation configuration uses:

~~~text
Input channels : 3
Output classes : 6
Tile size      : 512 × 512
Optimizer      : AdamW
Loss           : Dice + Cross-Entropy
Device         : CUDA when available
~~~

---

## Training

The segmentation pipeline supports:

- deterministic slide-level train/validation splitting
- real PANDA image/mask pairing
- class-aware tile sampling
- Dice + Cross-Entropy loss
- GPU training
- checkpoint generation
- independent evaluation

Train:

~~~bash
python scripts/train_real_panda.py
~~~

Evaluate a checkpoint:

~~~bash
python scripts/evaluate_panda_model.py \
    --checkpoint /mnt/d/PathoXData/experiments/panda/pathox_panda_sampler_best.pt
~~~

---

## Segmentation Results

Using the current class-aware tile sampling experiment:

~~~text
Validation tiles : 195

Mean Dice : 0.5093
Mean IoU  : 0.4379
~~~

Per-class performance is uneven, especially for very rare classes.

This result comes from a deliberately small experimental subset and should
not be interpreted as full PANDA performance, external validation, or clinical
performance.

---

## Native C++ / OpenMP Acceleration

Repeated tile-scoring operations are implemented in a native C++17 extension
and exposed to Python using pybind11.

OpenMP is used to parallelize the candidate-scoring loops.

### General tile scoring benchmark

~~~text
Python candidates : 1491
C++ candidates    : 1491

Python time : 0.9388 s
C++ time    : 0.0892 s

Speedup     : 10.53×
Exact match : True
~~~

### Tissue-aware candidate selection benchmark

Measured on a real WSI:

~~~text
Python time : 0.003388 s
C++ time    : 0.000311 s

Speedup               : 10.91×
Candidate count match : True
Max tissue error      : 3×10^-8
~~~

The native module handles performance-critical repeated operations while
Python remains responsible for pipeline orchestration and higher-level logic.

---

## End-to-End WSI Benchmark

The end-to-end benchmark measures the complete inference pipeline rather than
only the neural-network forward pass.

Configuration:

~~~text
GPU                 : NVIDIA GeForce RTX 3050 6GB Laptop GPU
Tile size           : 512 × 512
Stride              : 512
Thumbnail size      : 2048
Tissue threshold    : 0.12
Batch size          : 4
Uncertainty top-k   : 50
Warmup runs         : 1
Timed runs          : 3
~~~

### Results

~~~text
Mean end-to-end time : 10.4180 s
Std deviation        : 0.0650 s
Tiles analyzed       : 184
Throughput           : 17.66 tiles/s

Tissue fraction      : 0.1840
Non-background       : 0.8327
Mean confidence      : 0.7843
Mean entropy         : 0.3698
TTA disagreement     : 0.0063
~~~

The 17.66 tiles/s figure is an end-to-end pipeline throughput measurement
including:

- WSI I/O
- tissue screening
- tile extraction
- preprocessing
- GPU segmentation
- uncertainty processing
- whole-slide result generation

Run the benchmark:

~~~bash
python scripts/benchmark_end_to_end.py
~~~

Detailed results are written to:

~~~text
outputs/panda_benchmark/benchmark.json
~~~

---

## Whole-Slide Inference

PathoX can process an entire WSI by:

1. Opening the slide through OpenSlide.
2. Creating a low-resolution thumbnail.
3. Detecting tissue regions.
4. Generating candidate tile coordinates.
5. Filtering background regions.
6. Extracting 512×512 patches.
7. Running GPU segmentation in batches.
8. Aggregating class probabilities.
9. Computing confidence and entropy.
10. Applying TTA to selected difficult tiles.
11. Generating quantitative slide-level outputs.

Run:

~~~bash
python scripts/run_slide_inference.py
~~~

Outputs include:

~~~text
CSV predictions
Class distributions
Whole-slide heatmaps
Uncertainty heatmaps
Hard-tile outputs
JSON summary report
~~~

---

## Uncertainty-Aware Inference

PathoX uses test-time augmentation on selected difficult tiles.

The model evaluates:

~~~text
Original
Horizontal flip
Vertical flip
Horizontal + vertical flip
~~~

The predictions are averaged to obtain a more stable probability estimate.

The engine computes:

- normalized predictive entropy
- mean confidence
- TTA disagreement
- hard-tile rankings

This allows the system to surface uncertain regions rather than returning
only a hard segmentation map.

---

## Dashboard

PathoX includes an interactive Streamlit dashboard.

Launch locally:

~~~bash
streamlit run app/pathox_dashboard.py
~~~

Open:

~~~text
http://localhost:8501
~~~

The dashboard provides:

- WSI selection
- checkpoint selection
- tile-size configuration
- tissue threshold configuration
- uncertainty controls
- segmentation statistics
- class distribution
- whole-slide overview
- lesion/non-background heatmap
- uncertainty heatmap
- hard-tile inspection
- raw inference report

---

## Docker Deployment

The application is containerized with Docker and Docker Compose.

Build:

~~~bash
docker compose build
~~~

Start:

~~~bash
docker compose up -d
~~~

Open:

~~~text
http://localhost:8501
~~~

Stop:

~~~bash
docker compose down
~~~

The container is configured for NVIDIA GPU access.

The external pathology dataset is mounted read-only, while inference outputs are
written to the host output directory.

---

## Testing

Run the complete test suite:

~~~bash
pytest -q
~~~

Current status:

~~~text
39 passed
~~~

The tests cover areas including:

- WSI metadata and region access
- tile extraction
- tissue detection
- stain normalization
- segmentation components
- PANDA dataset utilities
- slide-level splitting
- uncertainty computation
- native C++ integration
- native tissue scoring
- inference pipeline behavior

---

## Reproducibility

The repository separates source code from large experimental artifacts.

Tracked:

~~~text
Source code
C++ extension
Training scripts
Inference scripts
Tests
Configuration
Docker files
Documentation
~~~

Excluded from Git:

~~~text
Raw pathology slides
Segmentation masks
Trained checkpoints
Generated inference outputs
Virtual environments
Native build artifacts
~~~

External data locations are configurable in:

~~~text
src/pathox/config.py
~~~

---

## Engineering Design

### 1. Pyramid-aware WSI access

PathoX works directly with the multi-resolution image pyramid exposed by
OpenSlide instead of loading a complete slide into memory.

### 2. Tissue-first computation

Low-resolution tissue screening eliminates large background regions before
expensive tile-level model inference.

### 3. Native acceleration

Repeated candidate-scoring operations are moved into C++17 and parallelized
with OpenMP.

### 4. GPU inference

Tile batches are processed using CUDA-enabled PyTorch.

### 5. Uncertainty estimation

TTA, entropy, confidence and disagreement metrics provide additional
information beyond a hard prediction.

### 6. Whole-slide aggregation

Local tile predictions are converted into slide-level class distributions,
spatial heatmaps and summary reports.

### 7. Containerized deployment

The complete application, Python dependencies and native extension can be
packaged into a reproducible Docker runtime.

---

## Performance Summary

~~~text
┌──────────────────────────────────────┬───────────────┐
│ Metric                               │ Result        │
├──────────────────────────────────────┼───────────────┤
│ Native tile scoring speedup          │ 10.53×        │
│ Native tissue scoring speedup        │ 10.91×        │
│ End-to-end WSI inference             │ 10.418 s      │
│ End-to-end throughput                │ 17.66 tiles/s │
│ WSI tiles analyzed                   │ 184           │
│ Validation Dice                      │ 0.5093        │
│ Validation IoU                       │ 0.4379        │
│ Automated tests                      │ 39 passed     │
└──────────────────────────────────────┴───────────────┘
~~~

---

## Limitations

PathoX is an engineering prototype and research demonstration.

The current quantitative evaluation is based on a small Radboud subset rather
than the complete PANDA dataset.

The current model therefore should not be considered:

- clinically validated
- production-ready diagnostic software
- representative of complete PANDA performance
- externally validated across institutions or scanners

Rare segmentation classes remain difficult in the current experiment.

Broader claims would require larger-scale training, stronger validation,
external datasets, improved architectures, calibration analysis, and
additional robustness testing.

---

## Future Extensions

Potential extensions include:

- larger-scale PANDA training
- stronger segmentation backbones
- multi-scale WSI inference
- asynchronous CPU/GPU pipelines
- CUDA kernels for additional preprocessing
- ONNX/TensorRT deployment
- patch-level lesion detection
- slide-level cancer classification
- interactive ROI analysis
- distributed slide processing
- model calibration
- external-dataset evaluation

---

## Why PathoX

PathoX is designed as a systems-oriented computer-vision project rather than
only a model-training experiment.

It combines:

~~~text
Digital Pathology
       +
Whole-Slide Image Processing
       +
Computer Vision
       +
Deep Learning
       +
C++ / OpenMP Optimization
       +
GPU Computing
       +
Uncertainty Estimation
       +
Deployment
~~~

The resulting system demonstrates the complete path from a very large medical
image to an optimized, uncertainty-aware, deployable AI inference pipeline.

---

## License

This project is intended for educational, research, and engineering
demonstration purposes.

---

## Author

**Hetram Gugrwal**

PathoX was developed as a systems-oriented computer-vision project combining
digital pathology, deep learning, native C++ optimization, GPU inference,
uncertainty estimation, and deployable inference infrastructure.
