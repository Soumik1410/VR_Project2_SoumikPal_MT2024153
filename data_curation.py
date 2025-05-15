import os
import json
import time
import csv
import re
from PIL import Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyCISj-f2EHTYXDEY7MqaHnTXopoQthUvfY")
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
image_folder = r"C:\IIITB MTech Sem 2\VR\VR Project 2\ABO Dataset\abo-images-small\images\small\00"
csv_metadata_path = r"C:\IIITB MTech Sem 2\VR\VR Project 2\ABO Dataset\abo-images-small\images\metadata\images.csv"
json_output_file = "vqa_results.json"
csv_output_file = "vqa_results.csv"
prompts = [
    "You are a VQA system. Given an image and metadata, generate an EASY multiple-choice question grounded in visible content. Return a JSON with: question, option_1, option_2, option_3, option_4, correct_option (as 'option_X').",
    "You are a VQA system. Given an image and metadata, generate a HARD multiple-choice question that focuses on a specific part, attribute or feature. Make sure it's grounded in visible content and return a JSON with: question, option_1, option_2, option_3, option_4, correct_option (as 'option_X')."
]


def load_csv_metadata(csv_path):
    metadata_dict = {}
    with open(csv_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata_dict[row['path']] = row
    return metadata_dict

def find_metadata_for_image(image_filename, metadata_dict):
    pattern = re.compile(rf"^[a-zA-Z0-9]{{2}}/{re.escape(image_filename)}$")
    for path_key in metadata_dict:
        if pattern.match(path_key):
            return {
                "width": metadata_dict[path_key].get("width"),
                "height": metadata_dict[path_key].get("height")
            }
    return None

def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

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
        cleaned = re.sub(r"^```json|^```|```$", "", response_text.strip(), flags=re.MULTILINE).strip()
        return json.loads(cleaned)
    except:
        return {"error": "Invalid JSON", "raw": response_text}

csv_metadata = load_csv_metadata(csv_metadata_path)


results = []
csv_rows = []


image_files = [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    metadata = find_metadata_for_image(img_file, csv_metadata)
    #print(metadata)

    if not metadata:
        print(f"Skipping {img_file}: no matching CSV metadata found.")
        continue

    image_bytes = load_image_bytes(img_path)

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

            required_keys = ["question", "option_1", "option_2", "option_3", "option_4", "correct_option"]
            if all(k in parsed for k in required_keys):
                csv_rows.append([
                    parsed["question"],
                    parsed["option_1"],
                    parsed["option_2"],
                    parsed["option_3"],
                    parsed["option_4"],
                    parsed["correct_option"]
                ])
        except Exception as e:
            print(f"Error on {img_file}: {e}")

        time.sleep(4)


with open(json_output_file, "w") as jf:
    json.dump(results, jf, indent=2)

with open(csv_output_file, "w", newline='', encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["question", "option_1", "option_2", "option_3", "option_4", "correct_option"])
    writer.writerows(csv_rows)

print(f"Saved {len(results)} responses to {json_output_file}")
print(f"Saved {len(csv_rows)} valid QAs to {csv_output_file}")
