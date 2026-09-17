from pathlib import Path

from pathox import TileExtractor, WSIReader


WSI_PATH = Path("data/test/CMU-1-Small-Region.svs")
OUTPUT_DIR = Path("outputs/test/tiles")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with WSIReader(WSI_PATH) as reader:
        extractor = TileExtractor(
            reader,
            tile_size=512,
            overlap=128,
        )

        tiles = list(
            extractor.iter_tiles(
                level=0,
                include_partial=True,
            )
        )

        print(f"Generated tile locations: {len(tiles)}")

        for index, tile in enumerate(tiles[:10]):
            image = extractor.read_tile(tile)
            path = OUTPUT_DIR / f"tile_{index:03d}.png"
            image.save(path)

        print("Saved first 10 tiles successfully.")


if __name__ == "__main__":
    main()
