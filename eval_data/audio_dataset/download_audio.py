import pandas as pd
import os
import subprocess

# Path to your CSV file
csv_file = "./audiocaps/dataset/val.csv"

# Output directory
output_dir = "audio_downloads"
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
data = pd.read_csv(csv_file)

data['success'] = False

# List to store successful downloads
successful_downloads = []

# Loop through the rows of the DataFrame
for index, row in data.iterrows():
    print(index+1, len(data))
    youtube_id = row['youtube_id']
    start_offset = row['start_time']
    audiocap_id = row['audiocap_id']  # Use this column for naming files
    duration = 10  # Fixed duration

    # Construct the video URL
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Output file name using audiocap_id and saving in WAV format
    output_file = os.path.join(output_dir, f"{audiocap_id}.wav")

    # yt-dlp command for best audio quality
    command = [
        "yt-dlp", url,
        "-f", "bestaudio/best",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg_i:-ss {start_offset} -t {duration}",
        "-x", "--audio-format", "wav",  # Set audio format to WAV
        "-o", output_file
    ]

    print(f"Processing audio for {youtube_id} (audiocap_id: {audiocap_id}) starting at {start_offset}s...")
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully downloaded: {output_file}")

        # Add to successful list and update 'success' column
        data.at[index, 'success'] = True

    except subprocess.CalledProcessError as e:
        print(f"Failed to process {youtube_id}: {e.stderr.decode().strip()}")

# Save updated CSV with 'success' column
data.to_csv(csv_file.replace('.csv', '_status.csv'), index=False)

print("All downloads processed, and 'success' column added to 'updated_downloads.csv'.")

