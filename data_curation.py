import os
import json
import time
import csv
from PIL import Image
import google.generativeai as genai

# Setup Gemini
genai.configure(api_key="AIzaSyCISj-f2EHTYXDEY7MqaHnTXopoQthUvfY")
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# Paths
image_folder = "path/to/abo/images"
metadata_folder = "path/to/abo/json"
json_output_file = "vqa_results.json"
csv_output_file = "vqa_results.csv"

# Prompts
prompts = [
    "You are a VQA system. Given an image and metadata, generate an EASY multiple-choice question grounded in visible content. Return a JSON with: question, option_1, option_2, option_3, option_4, correct_option (as 'option_X').",
    "You are a VQA system. Given an image and metadata, generate a HARD multiple-choice question that focuses on a specific part, attribute or feature. Make sure it's grounded in visible content and return a JSON with the same structure."
]

# Helpers
def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        return json.load(f)

def call_gemini(image_bytes, metadata, prompt):
    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [prompt]},
            {"role": "user", "parts": [{"mime_type": "image/png", "data": image_bytes}]},
            {"role": "user", "parts": [f"Metadata: {json.dumps(metadata)}"]}
        ]
    )
    return response.text

def parse_json_response(response_text):
    try:
        return json.loads(response_text)
    except:
        return {"error": "Invalid JSON", "raw": response_text}

# Initialize result containers
results = []
csv_rows = []

# Scan image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    meta_path = os.path.join(metadata_folder, os.path.splitext(img_file)[0] + ".json")

    if not os.path.exists(meta_path):
        print(f"Skipping {img_file}: no metadata found.")
        continue

    image_bytes = load_image_bytes(img_path)
    metadata = load_metadata(meta_path)

    for prompt in prompts:
        difficulty = "easy" if prompt == prompts[0] else "hard"
        print(f"Processing {img_file} with prompt: {difficulty}")
        try:
            response_text = call_gemini(image_bytes, metadata, prompt)
            parsed = parse_json_response(response_text)

            results.append({
                "image": img_file,
                "difficulty": difficulty,
                "qa": parsed
            })

            # If the response is valid and contains required fields, add to CSV
            if all(k in parsed for k in ["question", "option_1", "option_2", "option_3", "option_4"]):
                csv_rows.append([
                    parsed["question"],
                    parsed["option_1"],
                    parsed["option_2"],
                    parsed["option_3"],
                    parsed["option_4"]
                ])
        except Exception as e:
            print(f"Error on {img_file}: {e}")

        time.sleep(5)  # To avoid hitting Gemini Flash RPM limits

# Save JSON
with open(json_output_file, "w") as jf:
    json.dump(results, jf, indent=2)

# Save CSV
with open(csv_output_file, "w", newline='', encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["question", "option_1", "option_2", "option_3", "option_4"])
    writer.writerows(csv_rows)

print(f"Saved {len(results)} responses to {json_output_file}")
print(f"Saved {len(csv_rows)} valid QAs to {csv_output_file}")
