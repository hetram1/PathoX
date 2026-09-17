from pathlib import Path
import csv

from PIL import ImageDraw

from pathox import TileExtractor, TissueTileFilter, WSIReader


WSI_PATH = Path("data/test/CMU-1-Small-Region.svs")
OUTPUT_DIR = Path("outputs/test/tissue_filter")


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

        tile_filter = TissueTileFilter(
            reader,
            thumbnail_width=800,
            min_tissue_fraction=0.10,
        )

        scores = tile_filter.score_tiles(tiles)

        candidates = [
            score for score in scores
            if score.is_candidate
        ]

        csv_path = OUTPUT_DIR / "tile_scores.csv"

        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "x",
                    "y",
                    "level",
                    "width",
                    "height",
                    "tissue_fraction",
                    "candidate",
                ]
            )

            for score in scores:
                tile = score.tile
                writer.writerow(
                    [
                        tile.x,
                        tile.y,
                        tile.level,
                        tile.width,
                        tile.height,
                        f"{score.tissue_fraction:.6f}",
                        score.is_candidate,
                    ]
                )

        overlay = tile_filter.thumbnail.copy()
        draw = ImageDraw.Draw(overlay)

        for score in candidates:
            x0, y0, x1, y1 = tile_filter._thumbnail_bounds(
                score.tile
            )

            draw.rectangle(
                (x0, y0, x1, y1),
                outline=(255, 0, 0),
                width=2,
            )

        overlay_path = OUTPUT_DIR / "candidate_tiles.png"
        overlay.save(overlay_path)

        total = len(scores)
        selected = len(candidates)
        reduction = 1.0 - (selected / total)

        print("=== PathoX Tissue-Aware Tile Filtering ===")
        print(f"Total tiles: {total}")
        print(f"Candidate tiles: {selected}")
        print(f"Background skipped: {total - selected}")
        print(f"Compute reduction: {reduction:.2%}")
        print(f"Tile scores: {csv_path}")
        print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
