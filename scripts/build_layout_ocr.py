import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image

CSV_PATH = Path("data/sample_ocr_layout.csv")
IMAGE_PATH = Path("data/sample_poster.jpg")
OUTPUT_PATH = Path("data/sample_layout.json")


def read_words():
    words = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            text = row["text"].strip()
            if not text:
                continue

            words.append(
                {
                    "text": text,
                    "confidence": float(row["confidence"]),
                    "left": int(row["left"]),
                    "top": int(row["top"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "block_num": int(row["block_num"]),
                    "par_num": int(row["par_num"]),
                    "line_num": int(row["line_num"]),
                    "word_num": int(row["word_num"]),
                }
            )
    return words


def group_into_bands(words, image_width, image_height):
    median_height = statistics.median(word["height"] for word in words)
    band_tolerance = max(round(image_height * 0.015), round(median_height))
    column_gap = max(round(image_width * 0.02), round(median_height * 1.25))

    bands = []
    for word in sorted(words, key=lambda item: (item["top"], item["left"])):
        if not bands or word["top"] - bands[-1]["top"] > band_tolerance:
            bands.append({"top": word["top"], "words": []})
        bands[-1]["words"].append(word)

    regions = []
    for band in bands:
        horizontal_groups = []
        for word in sorted(band["words"], key=lambda item: (item["left"], item["top"])):
            word_right = word["left"] + word["width"]
            if (
                not horizontal_groups
                or word["left"] - horizontal_groups[-1]["right"] > column_gap
            ):
                horizontal_groups.append({"right": word_right, "words": [word]})
            else:
                horizontal_groups[-1]["right"] = max(
                    horizontal_groups[-1]["right"], word_right
                )
                horizontal_groups[-1]["words"].append(word)

        regions.extend(group["words"] for group in horizontal_groups)

    return regions


def build_region(region_words, region_id):
    left = min(word["left"] for word in region_words)
    top = min(word["top"] for word in region_words)
    right = max(word["left"] + word["width"] for word in region_words)
    bottom = max(word["top"] + word["height"] for word in region_words)

    lines = defaultdict(list)
    for word in region_words:
        line_key = (word["block_num"], word["par_num"], word["line_num"])
        lines[line_key].append(word)

    ordered_lines = sorted(
        lines.values(),
        key=lambda line: (
            min(word["top"] for word in line),
            min(word["left"] for word in line),
            line[0]["block_num"],
            line[0]["par_num"],
            line[0]["line_num"],
        ),
    )
    text = "\n".join(
        " ".join(
            word["text"]
            for word in sorted(line, key=lambda item: (item["left"], item["word_num"]))
        )
        for line in ordered_lines
    )

    return {
        "region_id": region_id,
        "bounding_box": {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        },
        "text": text,
        "average_confidence": round(
            statistics.mean(word["confidence"] for word in region_words), 2
        ),
        "number_of_words": len(region_words),
        "words": [
            {
                "text": word["text"],
                "confidence": word["confidence"],
                "bounding_box": {
                    "left": word["left"],
                    "top": word["top"],
                    "width": word["width"],
                    "height": word["height"],
                },
                "block_num": word["block_num"],
                "par_num": word["par_num"],
                "line_num": word["line_num"],
                "word_num": word["word_num"],
            }
            for word in sorted(
                region_words,
                key=lambda item: (item["top"], item["left"], item["word_num"]),
            )
        ],
    }


def main():
    with Image.open(IMAGE_PATH) as image:
        image_dimensions = {"width": image.width, "height": image.height}

    words = read_words()
    grouped_words = group_into_bands(
        words, image_dimensions["width"], image_dimensions["height"]
    )
    regions = [
        build_region(group, f"region_{index:03d}")
        for index, group in enumerate(
            sorted(
                grouped_words,
                key=lambda group: (
                    min(word["top"] for word in group),
                    min(word["left"] for word in group),
                ),
            ),
            start=1,
        )
    ]

    result = {"image_dimensions": image_dimensions, "regions": regions}
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(
        f"Saved {len(regions)} regions to {OUTPUT_PATH} "
        f"for image {image_dimensions['width']}x{image_dimensions['height']}."
    )
    for region in regions:
        box = region["bounding_box"]
        print(
            f"{region['region_id']}: "
            f"left={box['left']}, top={box['top']}, "
            f"width={box['width']}, height={box['height']}"
        )


if __name__ == "__main__":
    main()
