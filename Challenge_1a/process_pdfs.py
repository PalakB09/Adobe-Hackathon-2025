import os
import json
import fitz  # PyMuPDF
import jsonschema
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# Load spaCy model
import spacy
nlp = spacy.load("en_core_web_sm")

# Load schema
with open("./sample_dataset/schema/output_schema.json", "r") as f:
    output_schema = json.load(f)


def extract_title(page):
    blocks = page.get_text("dict")["blocks"]
    title_candidates = []

    for block in blocks:
        for line in block.get("lines", []):
            line_text = " ".join(span["text"].strip() for span in line["spans"])
            if len(line_text) < 5 or not any(c.isalnum() for c in line_text):
                continue
            avg_size = sum(span["size"] for span in line["spans"]) / len(line["spans"])
            title_candidates.append((avg_size, line_text.strip()))

    if not title_candidates:
        return "Untitled"

    return max(title_candidates, key=lambda x: x[0])[1]


def determine_heading_level(size, body_size):
    delta = size - body_size
    if delta > 4:
        return "H1"
    elif delta > 2.5:
        return "H2"
    elif delta > 1.5:
        return "H3"
    elif delta > 1:
        return "H4"
    return None


def parse_headings(page, page_number, body_size, recurring_texts, doc_title):
    blocks = page.get_text("dict")["blocks"]
    headings = []
    seen = set()

    for block in blocks:
        for line in block.get("lines", []):
            full_text = " ".join(span["text"].strip() for span in line["spans"]).strip()

            if not full_text or full_text in seen or full_text in recurring_texts or len(full_text) > 150:
                continue

            # Exclude title line
            if full_text.strip().lower() == doc_title.strip().lower():
                continue

            avg_size = sum(span["size"] for span in line["spans"]) / len(line["spans"])
            level = determine_heading_level(avg_size, body_size)

            if level:
                headings.append({
                    "level": level,
                    "text": full_text,
                    "page": page_number
                })
                seen.add(full_text)

    return headings


def estimate_body_font_size(doc):
    font_size_counts = Counter()
    for page in doc:
        for block in page.get_text("dict")['blocks']:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span['text'].strip()
                    if text:
                        font_size_counts.update({round(span['size'], 1): len(text)})
    return font_size_counts.most_common(1)[0][0] if font_size_counts else 12


def get_recurring_texts(doc):
    line_counts = Counter()
    for page in doc:
        for block in page.get_text("dict")['blocks']:
            for line in block.get('lines', []):
                line_text = "".join(span['text'] for span in line['spans']).strip()
                if line_text:
                    line_counts.update([line_text])
    return {text for text, count in line_counts.items() if count > 2}


def process_single_pdf(pdf_path):
    output_dir = Path("/app/output")
    try:
        doc = fitz.open(pdf_path)
        body_size = estimate_body_font_size(doc)
        recurring_texts = get_recurring_texts(doc)
        title = extract_title(doc[0])

        output_data = {
            "title": title,
            "outline": []
        }

        for page_num, page in enumerate(doc):
            headings = parse_headings(page, page_num, body_size, recurring_texts, title)
            output_data["outline"].extend(headings)

        jsonschema.validate(instance=output_data, schema=output_schema)

        output_file = output_dir / f"{pdf_path.stem}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ {pdf_path.name} processed successfully")
        return True

    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {e}")
        return False


def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(process_single_pdf, pdf_files)

if __name__ == "__main__":
    print("🚀 Starting PDF processing")
    process_pdfs()
    print("✅ All PDFs processed")
