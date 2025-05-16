import os
import pandas as pd
import shutil

csv_input_dir = r"C:\IIITB MTech Sem 2\VR\VR Project 2\Subset"
base_image_path = r"C:\IIITB MTech Sem 2\VR\VR Project 2\ABO Dataset\abo-images-small\images\small"
output_combined_csv = r"C:\IIITB MTech Sem 2\VR\VR Project 2\Subset\combined_vqa_single_answer.csv"
output_image_folder = r"C:\IIITB MTech Sem 2\VR\VR Project 2\Subset\images"

csv_files = [f for f in os.listdir(csv_input_dir) if f.endswith(".csv")]
combined_df = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(csv_input_dir, file)
    try:
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {file}: {e}")


combined_df.to_csv(output_combined_csv, index=False)
print(f"Combined CSV saved to: {output_combined_csv}")

missing_count = 0
copied_count = 0

for rel_path in combined_df["image_path"]:
    full_src_path = os.path.join(base_image_path, rel_path)

    dest_subdir = os.path.join(output_image_folder, os.path.dirname(rel_path))
    os.makedirs(dest_subdir, exist_ok=True)

    dest_path = os.path.join(dest_subdir, os.path.basename(rel_path))

    try:
        shutil.copy2(full_src_path, dest_path)
        copied_count += 1
    except Exception as e:
        print(f"Could not copy: {full_src_path} — {e}")
        missing_count += 1

print(f"Image copying complete: {copied_count} copied, {missing_count} missing.")
