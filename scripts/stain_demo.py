from pathlib import Path

from pathox import MacenkoNormalizer, WSIReader


WSI_PATH = Path("data/test/CMU-1-Small-Region.svs")
OUTPUT_DIR = Path("outputs/test/stain")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with WSIReader(WSI_PATH) as reader:
        thumbnail = reader.thumbnail(800)

    normalizer = MacenkoNormalizer()
    result = normalizer.normalize(thumbnail)

    input_path = OUTPUT_DIR / "input.png"
    output_path = OUTPUT_DIR / "macenko.png"

    thumbnail.save(input_path)
    result.image.save(output_path)

    print("=== PathoX Stain Normalization ===")
    print(f"Input size: {result.input_shape}")
    print(f"Output size: {result.output_shape}")
    print(f"Input: {input_path}")
    print(f"Normalized: {output_path}")


if __name__ == "__main__":
    main()
