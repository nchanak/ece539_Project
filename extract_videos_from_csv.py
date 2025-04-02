import csv
import os
import subprocess
import argparse


# To download zip, simply use wget https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part0.zip
# --csv is location of the csv that contains mappings for zip
# --zip_base is the base directory where the zip files are stored
# --target is the directory where the videos will be extracted
# --start is the index in the csv to start from
# --max is the maximum number of videos to extract
#
# Example usage:
#
#  python extract_videos_from_csv.py \
#  --csv video_mappings/OpenVid_part0.csv \
#  --zip_base . \
#  --target ./raw_videos \
#  --start 0 \
#  --max 10
def extract_videos(csv_path, zip_base_path, target_folder, start_index=0, max_videos=None):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Slice based on start_index and max_videos
    selected_rows = rows[start_index:]
    if max_videos is not None:
        selected_rows = selected_rows[:max_videos]

    for row in selected_rows:
        zip_file = os.path.join(zip_base_path, row['zip_file'])
        video_path = row['video_path']

        if not os.path.exists(zip_file):
            print(f"[!] Zip file not found: {zip_file}")
            continue

        print(f"[+] Extracting {video_path} from {zip_file}...")
        try:
            subprocess.run(['unzip', '-j', zip_file, video_path, '-d', target_folder], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to extract {video_path} from {zip_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific videos from zip files using CSV mapping.")
    parser.add_argument('--csv', required=True, help="Path to CSV file (e.g., video_mappings/OpenVid_part0.csv)")
    parser.add_argument('--zip_base', default='.', help="Base directory where zip files are stored")
    parser.add_argument('--target', required=True, help="Target directory to extract videos to")
    parser.add_argument('--start', type=int, default=0, help="Index in CSV to start from")
    parser.add_argument('--max', type=int, default=None, help="Maximum number of videos to extract (optional)")

    args = parser.parse_args()

    os.makedirs(args.target, exist_ok=True)

    extract_videos(args.csv, args.zip_base, args.target, args.start, args.max)
