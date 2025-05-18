import os
import json
import time
import csv
import re
from PIL import Image
import google.generativeai as genai


genai.configure(api_key="token string") #Removed actual token used for security
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
image_folder = r"C:\IIITB MTech Sem 2\VR\VR Project 2\ABO Dataset\abo-images-small\images\small\d4"
csv_metadata_path = r"C:\IIITB MTech Sem 2\VR\VR Project 2\ABO Dataset\abo-images-small\images\metadata\images.csv"
json_metadata_dir = r"C:\IIITB MTech Sem 2\VR\VR Project 2\ABO Dataset\abo-listings\listings\listings"
json_output_file = "vqa_single_answer_results_soumik_d4.json"
csv_output_file = "vqa_single_answer_results_soumik_d4.csv"

# Prompts
prompts = [
    "You are a VQA system. Given an image and metadata, generate an EASY question grounded in visible content. Return a JSON with: question and answer (answer should be a single word only).",
    "You are a VQA system. Given an image and metadata, generate a HARD question focusing on a specific part or feature of the image. Return a JSON with: question and answer (answer should be a single word only)."
]

# Load CSV metadata: maps image path -> image_id
def load_csv_metadata(csv_path):
    metadata_dict = {}
    with open(csv_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata_dict[row['path']] = {
                "image_id": row['image_id'],
                "path": row['path']
            }
    return metadata_dict

# Find CSV row by matching subdir/image.png format
def find_image_id_and_path(img_file, metadata_dict):
    pattern = re.compile(rf"^[a-zA-Z0-9]{{2}}/{re.escape(img_file)}$")
    for path_key in metadata_dict:
        if pattern.match(path_key):
            return metadata_dict[path_key]
    return None

# Load and combine multiple listings JSON files
def load_combined_metadata(json_dir):
    combined_metadata = []
    for filename in os.listdir(json_dir):
        if filename.startswith("listings_") and filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    for line in f:
                        line = line.strip()
                        if line:
                            combined_metadata.append(json.loads(line))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    return combined_metadata


# Build image_id → rich_metadata dictionary
def build_metadata_lookup(json_metadata):
    lookup = {}
    for item in json_metadata:
        main_id = item.get("main_image_id")
        if main_id:
            lookup[main_id] = item
        for other_id in item.get("other_image_id", []):
            if other_id:
                lookup[other_id] = item
    return lookup

# Load image as byte array
def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

# Call Gemini with image + metadata
def call_gemini(image_bytes, metadata, prompt):
    try:
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": [prompt]},
                {"role": "user", "parts": [{"mime_type": "image/png", "data": image_bytes}]},
                {"role": "user", "parts": [f"Metadata: {json.dumps(metadata)}"]}
            ]
        )
        return response.text
    except Exception as e:
        return str(e)

# Parse model output
def parse_json_response(response_text):
    try:
        cleaned = re.sub(r"^```json|^```|```$", "", response_text.strip(), flags=re.MULTILINE).strip()
        return json.loads(cleaned)
    except:
        return {"error": "Invalid JSON", "raw": response_text}


csv_metadata = load_csv_metadata(csv_metadata_path)
json_metadata = load_combined_metadata(json_metadata_dir)
#print(len(json_metadata))
rich_metadata_lookup = build_metadata_lookup(json_metadata)
del json_metadata
#print(len(rich_metadata_lookup))

results = []
csv_rows = []

# Filter only images in folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]

MAX_GEMINI_CALLS = 990
gemini_call_count = 0

# Process each image
for img_file in image_files:
    if gemini_call_count >= MAX_GEMINI_CALLS:
        print(f"Reached limit of {MAX_GEMINI_CALLS} Gemini calls. Stopping.")
        break

    img_path = os.path.join(image_folder, img_file)
    base_info = find_image_id_and_path(img_file, csv_metadata)

    if not base_info:
        print(f"Skipping {img_file}: No image_id found in CSV.")
        continue

    print(base_info)
    image_id = base_info["image_id"]
    image_path = base_info["path"]

    rich_metadata = rich_metadata_lookup.get(image_id)

    if not rich_metadata:
        print(f"Skipping {img_file}: No matching metadata in JSON for image_id {image_id}")
        continue

    image_bytes = load_image_bytes(img_path)

    for prompt in prompts:
        if gemini_call_count >= MAX_GEMINI_CALLS:
            print(f"Reached limit of {MAX_GEMINI_CALLS} Gemini calls. Stopping.")
            break

        difficulty = "easy" if prompt == prompts[0] else "hard"
        print(f"[{gemini_call_count + 1} / 990] : Processing {img_file} ({image_id}) with prompt: {difficulty}")
        try:
            response_text = call_gemini(image_bytes, rich_metadata, prompt)
            parsed = parse_json_response(response_text)

            results.append({
                "image": img_file,
                "image_id": image_id,
                "difficulty": difficulty,
                "qa": parsed
            })

            if "question" in parsed and "answer" in parsed:
                csv_rows.append([
                    image_path,
                    parsed["question"],
                    parsed["answer"]
                ])
            gemini_call_count += 1
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            if "GenerateRequestsPerDayPerProjectPerModel" in str(e):
                gemini_call_count = MAX_GEMINI_CALLS
        time.sleep(4)

# Save to JSON
with open(json_output_file, "w", encoding="utf-8") as jf:
    json.dump(results, jf, indent=2)

# Save to CSV
with open(csv_output_file, "w", newline='', encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["image_path", "question", "answer"])
    writer.writerows(csv_rows)

print(f"Saved {len(results)} responses to {json_output_file}")
print(f"Saved {len(csv_rows)} valid QAs to {csv_output_file}")
