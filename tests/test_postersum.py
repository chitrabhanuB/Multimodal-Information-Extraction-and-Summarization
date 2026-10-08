try:
    from datasets import load_dataset
except ImportError as error:
    raise SystemExit(
        "The 'datasets' package is required. Install it with: "
        "python -m pip install datasets"
    ) from error


def main():
    dataset = load_dataset("rohitsaxena/PosterSum")
    test_split = dataset["test"]
    first_sample = test_split[0]

    print("Available splits:", list(dataset.keys()))
    print("Test samples:", len(test_split))
    print("Columns:", test_split.column_names)
    print("First sample title:", first_sample["title"])
    print("First sample abstract:", first_sample["abstract"])
    print("First sample conference:", first_sample["conference"])
    print("First sample year:", first_sample["year"])


if __name__ == "__main__":
    main()