from PIL import Image
import pytesseract

with Image.open("data/sample_poster.jpg") as image:
    ocr_data = pytesseract.image_to_data(
        image,
        lang="eng",
        output_type=pytesseract.Output.DATAFRAME,
    )

ocr_data = ocr_data.rename(columns={"conf": "confidence"})
ocr_data = ocr_data.loc[ocr_data["text"].fillna("").str.strip().ne("")].copy()

output_path = "data/sample_ocr_layout.csv"
ocr_data.to_csv(output_path, index=False)

print(f"Detected text elements: {len(ocr_data)}")
columns_to_show = ["text", "confidence", "left", "top", "width", "height"]
print(ocr_data[columns_to_show].head(15).to_string(index=False))
print(f"OCR layout data saved to {output_path}")
