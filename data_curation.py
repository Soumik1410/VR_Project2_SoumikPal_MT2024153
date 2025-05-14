import os
import json
import time
import base64
from PIL import Image
import google.generativeai as genai

# Setup
genai.configure(api_key="AIzaSyCISj-f2EHTYXDEY7MqaHnTXopoQthUvfY")
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# Paths
image_folder = "path/to/abo/images"
metadata_folder = "path/to/abo/json"
output_file = "vqa_results.json"

# Prompts
prompts = [
    "You are a VQA system. Given an image and metadata, generate an EASY multiple-choice question grounded in visible content. Return a JSON with: question, option_1, option_2, option_3, option_4, correct_option (as 'option_X').",
    "You are a VQA system. Given an image and metadata, generate a HARD multiple-choice question that requires more reasoning. Make sure it's grounded in visible content and return a JSON with the same structure."
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

# Processing loop
results = []
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
        print(f"Processing {img_file} with prompt: {'easy' if prompt == prompts[0] else 'hard'}")
        try:
            response_text = call_gemini(image_bytes, metadata, prompt)
            parsed = parse_json_response(response_text)
            results.append({
                "image": img_file,
                "difficulty": "easy" if prompt == prompts[0] else "hard",
                "qa": parsed
            })
        except Exception as e:
            print(f"Error on {img_file}: {e}")

        # Respect Gemini Flash free-tier limits (15 RPM = 1 req every 4 sec)
        time.sleep(5)

# Save all results
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved all results to {output_file}")
