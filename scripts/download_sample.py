from io import BytesIO
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image


def main():
    test_split = load_dataset("rohitsaxena/PosterSum", split="test")
    first_sample = test_split[0]
    image_url = first_sample["image_url"]

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    output_path = Path(__file__).resolve().parent / "data" / "sample_poster.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(BytesIO(response.content)) as image:
        image = image.convert("RGB")
        image.save(output_path, format="JPEG")
        dimensions = image.size

    print("Image URL:", image_url)
    print("Saved file:", output_path)
    print("Image dimensions:", dimensions)


if __name__ == "__main__":
    main()