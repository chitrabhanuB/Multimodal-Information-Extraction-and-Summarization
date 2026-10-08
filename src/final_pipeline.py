# Colab setup (run once in a notebook cell):
# !apt-get -qq update && apt-get -qq install -y tesseract-ocr
# !pip install -q datasets requests pillow pytesseract transformers accelerate qwen-vl-utils rouge-score

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "rohitsaxena/PosterSum"
OCR_CONFIDENCE_THRESHOLD = 60.0


def download_poster(image_url: str, output_path: str | Path) -> Path:
	"""Download a POSTERSUM image to a local path."""
	import requests

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	response = requests.get(image_url, timeout=60)
	response.raise_for_status()
	output_path.write_bytes(response.content)
	return output_path


def run_ocr(image_path: str | Path) -> list[dict[str, Any]]:
	"""Run English Tesseract OCR and retain word boxes and line identifiers."""
	import pytesseract

	result = pytesseract.image_to_data(
		str(image_path), lang="eng", output_type=pytesseract.Output.DICT
	)
	words = []
	for index, text in enumerate(result["text"]):
		words.append(
			{
				"text": str(text).strip(),
				"confidence": float(result["conf"][index]),
				"left": int(result["left"][index]),
				"top": int(result["top"][index]),
				"width": int(result["width"][index]),
				"height": int(result["height"][index]),
				"block_num": int(result["block_num"][index]),
				"par_num": int(result["par_num"][index]),
				"line_num": int(result["line_num"][index]),
				"word_num": int(result["word_num"][index]),
			}
		)
	return words


def filter_ocr(
	words: list[dict[str, Any]], threshold: float = OCR_CONFIDENCE_THRESHOLD
) -> list[dict[str, Any]]:
	"""Keep nonempty OCR words at or above the confidence threshold."""
	return [
		word.copy()
		for word in words
		if word["text"].strip() and word["confidence"] >= threshold
	]


def build_spatial_ocr(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Group words by Tesseract line and preserve word and line bounding boxes."""
	grouped_lines = defaultdict(list)
	for word in words:
		key = (word["block_num"], word["par_num"], word["line_num"])
		grouped_lines[key].append(word)

	spatial_lines = []
	for (block_num, par_num, line_num), line_words in grouped_lines.items():
		line_words.sort(key=lambda word: (word["left"], word["word_num"]))
		left = min(word["left"] for word in line_words)
		top = min(word["top"] for word in line_words)
		right = max(word["left"] + word["width"] for word in line_words)
		bottom = max(word["top"] + word["height"] for word in line_words)
		spatial_lines.append(
			{
				"block_num": block_num,
				"par_num": par_num,
				"line_num": line_num,
				"text": " ".join(word["text"] for word in line_words),
				"bounding_box": {
					"left": left,
					"top": top,
					"width": right - left,
					"height": bottom - top,
				},
				"words": line_words,
			}
		)

	return sorted(
		spatial_lines,
		key=lambda line: (
			line["bounding_box"]["top"],
			line["bounding_box"]["left"],
			line["block_num"],
			line["par_num"],
			line["line_num"],
		),
	)


def generate_summary(
	image_path: str | Path,
	spatial_ocr: list[dict[str, Any]],
	model: Any = None,
	processor: Any = None,
	model_id: str = MODEL_ID,
) -> str:
	"""Generate an image-grounded academic summary with Qwen2.5-VL."""
	import torch
	from qwen_vl_utils import process_vision_info
	from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

	if (model is None) != (processor is None):
		raise ValueError("Pass both model and processor, or neither.")

	if model is None:
		has_gpu = torch.cuda.is_available()
		print(f"CUDA available: {has_gpu}")
		if has_gpu:
			dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		else:
			dtype = torch.float32
		model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
			model_id, torch_dtype=dtype, device_map="auto"
		)
		processor = AutoProcessor.from_pretrained(model_id)

	prompt = (
		"Write an academic abstractive summary of approximately 120-180 words. "
		"Cover the research problem, methodology, contribution, and important findings. "
		"Do not invent information; omit details that are not visible or clearly supported. "
		"Use the poster image as the primary source. The OCR text below is supporting "
		"evidence and may contain recognition errors. Use its coordinates and line "
		"grouping to help locate text, but verify details against the image.\n\n"
		"Spatial OCR (each line includes a pixel bounding box):\n"
		f"{json.dumps(spatial_ocr, ensure_ascii=False)}"
	)
	messages = [
		{
			"role": "user",
			"content": [
				{"type": "image", "image": str(image_path)},
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

	generated_ids = model.generate(**inputs, max_new_tokens=320)
	answer_ids = generated_ids[:, inputs.input_ids.shape[1] :]
	return processor.batch_decode(
		answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)[0].strip()


def calculate_rouge(generated_summary: str, reference_abstract: str) -> dict[str, float]:
	"""Calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
	from rouge_score import rouge_scorer

	scorer = rouge_scorer.RougeScorer(
		["rouge1", "rouge2", "rougeL"], use_stemmer=True
	)
	scores = scorer.score(reference_abstract, generated_summary)
	return {name: round(score.fmeasure, 4) for name, score in scores.items()}


def run_pipeline(
	sample_index: int = 0,
	split: str = "test",
	dataset_id: str = DATASET_ID,
	model_id: str = MODEL_ID,
	model: Any = None,
	processor: Any = None,
) -> dict[str, Any]:
	"""Run POSTERSUM download, OCR, visual summarization, and ROUGE scoring."""
	from datasets import load_dataset

	dataset = load_dataset(dataset_id, split=split)
	sample = dataset[sample_index]
	reference_abstract = sample["abstract"]

	# Download the sample temporarily so the pipeline leaves no poster artifact.
	with tempfile.TemporaryDirectory(prefix="postersum-") as temporary_directory:
		image_path = download_poster(
			sample["image_url"], Path(temporary_directory) / "poster.jpg"
		)

		# Extract OCR, apply the final confidence threshold, and preserve geometry.
		raw_ocr = run_ocr(image_path)
		filtered_ocr = filter_ocr(raw_ocr)
		spatial_ocr = build_spatial_ocr(filtered_ocr)

		# Use the poster image as primary evidence and spatial OCR as support.
		summary = generate_summary(
			image_path, spatial_ocr, model=model, processor=processor, model_id=model_id
		)

	# Compare the generated summary with this POSTERSUM sample's abstract.
	rouge_scores = calculate_rouge(summary, reference_abstract)
	return {
		"title": sample.get("title", ""),
		"summary": summary,
		"reference_abstract": reference_abstract,
		"rouge": rouge_scores,
		"raw_ocr_word_count": len(raw_ocr),
		"filtered_ocr_word_count": len(filtered_ocr),
		"spatial_ocr": spatial_ocr,
	}


if __name__ == "__main__":
	result = run_pipeline()
	print("Title:", result["title"])
	print("OCR words:", result["raw_ocr_word_count"])
	print("Words at confidence >= 60:", result["filtered_ocr_word_count"])
	print("\nGenerated summary:\n", result["summary"])
	print("\nROUGE F1:", result["rouge"])
