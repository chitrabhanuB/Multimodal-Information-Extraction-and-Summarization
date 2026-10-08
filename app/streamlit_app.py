import json
from pathlib import Path
import sys
import tempfile

import streamlit as st
import torch
from PIL import Image

st.set_page_config(page_title="Multimodal Poster Information Extractor", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_pipeline import build_spatial_ocr, filter_ocr, run_ocr

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@st.cache_resource(show_spinner=False)
def load_qwen_model():
    """Load the Qwen model and processor once in a CUDA-enabled runtime."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    return model, processor


def generate_multimodal_summary(image_path: Path, spatial_lines: list[dict]) -> str:
    """Summarize the poster image with spatial OCR as supporting context."""
    from qwen_vl_utils import process_vision_info

    model, processor = load_qwen_model()
    with Image.open(image_path) as uploaded_image:
        qwen_image = uploaded_image.convert("RGB")
    qwen_image.thumbnail((768, 768), Image.Resampling.LANCZOS)

    prompt = (
        "Write a 120-180 word academic paragraph about this research poster. Cover "
        "the problem or motivation, methodology, contribution, and important findings. "
        "Use the poster image as the primary source and the spatially grouped text below "
        "as secondary supporting context. Use the image to correct obvious OCR errors. "
        "Do not invent facts or include claims unsupported by the poster. Do not mention "
        "OCR in the summary.\n\n"
        "Spatially grouped text and bounding boxes:\n"
        f"{json.dumps(spatial_lines, ensure_ascii=False)}"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        resized_path = Path(temporary_directory) / "poster_resized.jpg"
        qwen_image.save(resized_path, format="JPEG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(resized_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=192)
    answer_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


@st.cache_data(show_spinner=False)
def extract_ocr(image_bytes: bytes, suffix: str):
    """Run OCR once per uploaded image and retain word and line geometry."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_image = Path(temporary_directory) / f"poster{suffix}"
        temporary_image.write_bytes(image_bytes)
        raw_words = run_ocr(temporary_image)

    all_words = [word for word in raw_words if word["text"].strip()]
    high_confidence_words = filter_ocr(all_words, threshold=60)
    spatial_lines = build_spatial_ocr(high_confidence_words)
    return all_words, high_confidence_words, spatial_lines


# Provide the poster upload control.
st.title("Multimodal Poster Information Extractor")
cuda_available = torch.cuda.is_available()
if cuda_available:
    st.sidebar.success("GPU inference available")
else:
    st.sidebar.info("Local CPU mode — OCR only")

uploaded_file = st.file_uploader(
    "Upload a poster image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Upload a PNG, JPG, or JPEG poster to begin.")
else:
    image_bytes = uploaded_file.getvalue()
    poster = Image.open(uploaded_file).convert("RGB")

    # Show the poster while OCR runs, without saving a permanent upload.
    image_column, status_column = st.columns([1.5, 1])
    with image_column:
        st.image(poster, caption=uploaded_file.name, use_container_width=True)

    try:
        with status_column:
            with st.spinner("Running Tesseract OCR..."):
                all_words, high_confidence_words, spatial_lines = extract_ocr(
                    image_bytes, Path(uploaded_file.name).suffix or ".png"
                )
        st.success("OCR completed.")
    except Exception as error:
        st.error(
            "OCR could not be completed. Confirm Tesseract is installed and available "
            f"to pytesseract. Details: {error}"
        )
        st.stop()

    # Summarize OCR coverage with the requested counts.
    st.subheader("OCR Results")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Total OCR words", len(all_words))
    metric_columns[1].metric("High-confidence words (>= 60)", len(high_confidence_words))
    metric_columns[2].metric("Spatial OCR lines", len(spatial_lines))

    # Present line text and bounding boxes in reading order.
    with st.expander("Spatial OCR text", expanded=True):
        if spatial_lines:
            for line in spatial_lines:
                box = line["bounding_box"]
                st.text(
                    f"[{box['left']}, {box['top']}, {box['width']}, {box['height']}] "
                    f"{line['text']}"
                )
        else:
            st.info("No words met the confidence threshold for spatial line grouping.")

    # Show each recognized nonempty word, score, and pixel bounding box.
    with st.expander("Word-level OCR and bounding boxes"):
        word_rows = [
            {
                "text": word["text"],
                "confidence": word["confidence"],
                "high_confidence": word["confidence"] >= 60,
                "left": word["left"],
                "top": word["top"],
                "width": word["width"],
                "height": word["height"],
            }
            for word in all_words
        ]
        st.dataframe(word_rows, hide_index=True, use_container_width=True)

    # Run Qwen only after an explicit request and only when CUDA is available.
    st.subheader("Multimodal Summary")
    if st.button("Generate Summary"):
        if not cuda_available:
            st.info(
                "Multimodal summary generation requires a GPU-enabled environment such as Google Colab."
            )
        else:
            try:
                with st.spinner("Generating poster summary with Qwen2.5-VL..."):
                    with tempfile.TemporaryDirectory() as temporary_directory:
                        temporary_image = Path(temporary_directory) / (
                            f"poster{Path(uploaded_file.name).suffix or '.png'}"
                        )
                        temporary_image.write_bytes(image_bytes)
                        summary = generate_multimodal_summary(
                            temporary_image, spatial_lines
                        )
                st.markdown("### Generated Summary")
                st.write(summary)
            except Exception as error:
                st.error(f"Summary generation failed: {error}")
