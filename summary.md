# Multimodal Information Extraction and Summarization — Project Summary

## 1. Project Goal
This project builds a multimodal pipeline for extracting research information from academic poster images and generating a short academic summary grounded in the poster content.

The main objective is to:
- extract text from poster images using OCR,
- preserve spatial information (word and line locations),
- use the poster image as the primary evidence source together with OCR context,
- generate a summary with Qwen2.5-VL,
- evaluate the output against the POSTERSUM abstract using ROUGE metrics.

## 2. Problem Statement
The project targets research posters from the POSTERSUM dataset. The challenge is not just OCR extraction but also understanding visual text layout, identifying relevant content, and generating image-grounded summaries that are faithful to what is visible on the poster.

## 3. Core Methodology Implemented
The current pipeline follows this flow:

1. Load a sample from the `rohitsaxena/PosterSum` dataset.
2. Download the poster image from its `image_url`.
3. Run English Tesseract OCR.
4. Keep only non-empty OCR words with confidence >= 60.
5. Group filtered words into spatial OCR lines using Tesseract's block/paragraph/line metadata and bounding boxes.
6. Feed the poster image and spatial OCR text to Qwen2.5-VL-3B-Instruct.
7. Generate a 120–180 word academic summary.
8. Evaluate summary quality using ROUGE-1, ROUGE-2, and ROUGE-L against the dataset abstract.

This is implemented in the main pipeline file:
- `src/final_pipeline.py`

## 4. What Has Already Been Built

### 4.1 Main pipeline
`src/final_pipeline.py` contains the complete prototype pipeline with these functions:
- `download_poster()`
- `run_ocr()`
- `filter_ocr()`
- `build_spatial_ocr()`
- `generate_summary()`
- `calculate_rouge()`
- `run_pipeline()`

This file is the main research prototype logic. It downloads a poster, runs OCR, filters low-confidence words, groups spatial OCR lines, calls the multimodal model, and computes ROUGE scores.

### 4.2 Streamlit demo app
`app/streamlit_app.py` is a UI wrapper that lets a user:
- upload a poster image,
- run OCR and inspect word-level / line-level results,
- view spatial OCR bounding boxes,
- trigger Qwen multimodal summary generation if a GPU-enabled environment is available.

This app is user-facing and useful for demonstration and manual inspection.

### 4.3 Supporting scripts
The project also includes scripts for OCR-related preprocessing:
- `scripts/build_layout_ocr.py`
- `scripts/download_sample.py`

These help with dataset or layout preparation tasks and are part of the experimental pipeline setup.

### 4.4 Test scripts
There are validation / smoke-test scripts in `tests/`:
- `tests/test_ocr.py`
- `tests/test_ocr_layout.py`
- `tests/test_postersum.py`

These validate:
- OCR extraction from sample poster images,
- OCR output layout collection,
- access to the POSTERSUM dataset and metadata.

### 4.5 Notebook experiments
`notebooks/final_experiments.ipynb` contains the experimental notebook used for running the project logic and collecting the final results.

## 5. Current Dataset and Setup
The pipeline is designed around the Hugging Face dataset:
- `rohitsaxena/PosterSum`

It uses the dataset's `test` split by default and reads:
- `image_url`
- `title`
- `abstract`

The UI and pipeline expect the environment to support:
- Python
- Hugging Face Datasets
- Tesseract OCR via `pytesseract`
- Pillow
- PyTorch
- Transformers
- Qwen2.5-VL-3B-Instruct
- `rouge-score`

## 6. Dependencies Listed
The project dependencies are defined in `requirements.txt`:
- torch
- transformers>=4.49.0
- accelerate
- qwen-vl-utils
- datasets
- requests
- Pillow
- pytesseract
- rouge-score
- pandas
- streamlit

## 7. Data and Output Artifacts Already Produced
The project includes local artifacts in `data/` and `results/`:

### Data directory
`data/` contains:
- `sample_poster.jpg`
- `sample_ocr.txt`
- `sample_ocr_layout.csv`
- `sample_layout.json`

These files represent sample OCR and layout processing outputs.

### Results directory
`results/` contains:
- `final_poster_results.csv`
- `spatial_ocr_poster2_line_level.png`

The CSV stores results for three evaluated posters. The line-level visualization shows the spatial OCR grouping for Poster 2.

## 8. Evaluation Results Achieved
The project reports evaluation on 3 POSTERSUM test posters.

From the project README:
- average ROUGE-1 F1: 0.3862
- average ROUGE-2 F1: 0.1120
- average ROUGE-L F1: 0.1837

The actual per-poster evaluation results in `results/final_poster_results.csv` are:

| Poster | OCR Words | Filtered OCR | Spatial Lines | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 225 | 69 | 16 | 0.3645 | 0.1132 | 0.1589 |
| 2 | 462 | 314 | 22 | 0.5194 | 0.1951 | 0.2767 |
| 3 | 7 | 7 | 2 | 0.2747 | 0.0276 | 0.1154 |

Important observation:
- Poster 2 had relatively strong OCR quality and better result quality.
- Poster 3 had very poor OCR coverage (only 7 words), leading to worse summary quality.
- This confirms that OCR quality and poster layout heavily impact multimodal summarization performance.

## 9. Known Limitations Already Identified
The repo explicitly notes important limitations:
- Tesseract can misread stylized text, equations, and dense posters.
- Confidence filtering may remove useful words.
- OCR line grouping does not fully capture reading order or section structure.
- Qwen2.5-VL-3B-Instruct is resource-intensive and works best on GPU-enabled Colab.
- ROUGE scores measure lexical overlap, not factual correctness or deep semantic quality.

## 10. Project State / Current Status
At this point, the project is a functioning research prototype and demo system, not a production-grade solution.

What is already done:
- dataset integration,
- poster download,
- OCR extraction,
- confidence filtering,
- spatial line grouping,
- multimodal summary generation,
- ROSUGE evaluation,
- UI experimentation / demo,
- result collection and documentation.

What is still open / should be next focus:
- improving OCR robustness on dense or low-contrast posters,
- better layout understanding and reading order,
- stronger visual grounding or post-generation validation,
- improved summary quality evaluation beyond ROUGE,
- making the pipeline more reliable and generalizable.

## 11. How This Project Is Structured
Project structure:

```text
Multimodal-Information-Extraction/
├── README.md
├── requirements.txt
├── summary.md
├── src/
│   └── final_pipeline.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── build_layout_ocr.py
│   └── download_sample.py
├── tests/
│   ├── test_ocr.py
│   ├── test_ocr_layout.py
│   └── test_postersum.py
├── notebooks/
│   └── final_experiments.ipynb
├── data/
│   ├── sample_poster.jpg
│   ├── sample_ocr.txt
│   ├── sample_ocr_layout.csv
│   └── sample_layout.json
├── results/
│   ├── final_poster_results.csv
│   └── spatial_ocr_poster2_line_level.png
└── .gitignore
```

## 12. Recommended Next-Step Direction for ChatGPT
When passing this project to ChatGPT for the next steps, the key context to keep in mind is:

- This is a multimodal poster summarization prototype.
- The current pipeline is built and partially validated.
- OCR + layout extraction is a major bottleneck.
- Summary quality depends strongly on OCR fidelity and poster design.
- The main technical challenge is improving extraction and grounding quality before scaling evaluation.

A strong next-step agenda would be one of these:
1. improve OCR quality and spatial layout handling,
2. enhance the multimodal prompt / generation logic,
3. add a stronger validation strategy beyond ROUGE,
4. build a more robust production-ready automated pipeline.

## 13. Short Context Summary for Future AI Assistance
This project uses OCR + image-grounded Qwen multimodal generation to summarize academic posters. It already downloads posters, runs Tesseract, filters noise, groups OCR spatially, generates summaries, and evaluates them with ROUGE. The prototype is functional, but OCR quality and layout understanding remain the biggest issues affecting performance.

This is the state of the project as of now.
