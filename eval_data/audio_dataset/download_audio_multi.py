
import pandas as pd
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to your CSV file
filename = 'val'
csv_file = f"./audiocaps/dataset/{filename}.csv"

# Output directory
output_dir = f"{filename}_yid_audio_downloads"
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
data = pd.read_csv(csv_file)

# Initialize 'success' column with False values
data['success'] = False

'''
# Function to download audio for each row
def download_audio(row):
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
        # If the file already exists, yt-dlp will overwrite it by default
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully downloaded: {output_file}")

        # Mark as successful in the DataFrame's 'success' column
        data.at[row.name, 'success'] = True

    except subprocess.CalledProcessError as e:
        print(f"Failed to process {youtube_id}: {e.stderr.decode().strip()}")
        # Mark as unsuccessful in the 'success' column
        data.at[row.name, 'success'] = False
'''

def download_audio(row):
    youtube_id = row['youtube_id']
    start_offset = row['start_time']
    audiocap_id = row['audiocap_id']
    duration = 10
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    output_file = os.path.join(output_dir, f"{youtube_id}.wav")
    if os.path.exists(output_file): return


    command = [
        "yt-dlp", url,
        "-f", "bestaudio/best",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg_i:-ss {start_offset} -t {duration}",
        "-x", "--audio-format", "wav",
        "-o", output_file,
        "--verbose"
    ]
    print(f"{audiocap_id}")
    #print(f"Thread {os.getpid()} is processing audio for {youtube_id} (audiocap_id: {audiocap_id}) starting at {start_offset}s...")
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(output_file):
            print(f"Successfully downloaded: {output_file}")
            data.at[row.name, 'success'] = True
        else:
            print(f"Download failed for {audiocap_id}. File not found: {output_file}")
            data.at[row.name, 'success'] = False
    except subprocess.CalledProcessError as e:
        print(f"Error during download for {audiocap_id}: {e.stderr.decode().strip()}")
        data.at[row.name, 'success'] = False


# Use ThreadPoolExecutor to download multiple audios in parallel
'''with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your system's capability
    executor.map(download_audio, [row for _, row in data.iterrows()])
'''
with ThreadPoolExecutor(max_workers=16) as executor:
    future_to_row = {executor.submit(download_audio, row): row for _, row in data.iterrows()}
    for future in as_completed(future_to_row):
        row = future_to_row[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error with audiocap_id {row['audiocap_id']}: {e}")

# Save updated CSV with 'success' column
data.to_csv(csv_file.replace('.csv', '_status_yid.csv'), index=False)

print("All downloads processed, and 'success' column added to 'updated_downloads.csv'.")

