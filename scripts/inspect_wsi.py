from pathlib import Path

from pathox import WSIReader


WSI_PATH = Path("data/test/CMU-1-Small-Region.svs")
OUTPUT_DIR = Path("outputs/test")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with WSIReader(WSI_PATH) as reader:
        metadata = reader.metadata

        print("=== PathoX WSI Inspection ===")
        print(f"File: {metadata.path}")
        print(f"Vendor: {metadata.vendor}")
        print(f"Levels: {metadata.level_count}")
        print(f"Dimensions: {metadata.level_dimensions}")
        print(f"Downsamples: {metadata.level_downsamples}")
        print(f"MPP X: {metadata.mpp_x}")
        print(f"MPP Y: {metadata.mpp_y}")

        thumbnail = reader.thumbnail(800)
        thumbnail_path = OUTPUT_DIR / "thumbnail.png"
        thumbnail.save(thumbnail_path)

        print(f"Thumbnail: {thumbnail_path}")

        tile = reader.read_region(
            x=0,
            y=0,
            level=0,
            width=512,
            height=512,
        )

        tile_path = OUTPUT_DIR / "tile_0_0.png"
        tile.save(tile_path)

        print(f"Tile: {tile_path}")
        print(f"Tile size: {tile.size}")

        print("WSI inspection successful.")


if __name__ == "__main__":
    main()
