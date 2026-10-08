from PIL import Image
import pytesseract

with Image.open("data/sample_poster.jpg") as image:
    extracted_text = pytesseract.image_to_string(image, lang="eng")

print(extracted_text)

output_path = "data/sample_ocr.txt"
with open(output_path, "w", encoding="utf-8") as output_file:
    output_file.write(extracted_text)

print(f"OCR result saved to {output_path}")
