import subprocess
import os
from pathlib import Path
from datasets import load_dataset, Audio
import numpy as np
import librosa
from tqdm import tqdm

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='/tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    # --quiet --no-warnings
    # /home/dominik/anaconda3/bin/yt-dlp
    command = f"""
        yt-dlp -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" --force-keyframes-at-cuts {url_base}{video_identifier}
    """.strip()
    print(command
          )
    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            print(err)
            print(err.output)
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def load_musiccaps(
    data_dir: str,
    sampling_rate: int = 16000,
    limit: int = None,
    num_proc: int = 8,
    writer_batch_size: int = 1000,
    return_without_audio=False
):
    """
    Download the clips within the MusicCaps dataset from YouTube.
    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    ds = load_dataset('google/MusicCaps', split='train')
    
    if return_without_audio:
        return ds
    
    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))


if __name__ == "__main__":
    ds = load_musiccaps(
    '/Users/alexandrasouly/code/muscaps/data/datasets/muscaps/audio',
    sampling_rate=16000,
    limit=None,
    num_proc=8,
    writer_batch_size=1000,
    return_without_audio=False
)
    audio_dir = "data/datasets/muscaps/audio"
    for audio_file in tqdm(os.listdir(audio_dir)):
        if audio_file[-3:] == 'wav':
            audio_path = os.path.join(audio_dir, audio_file)
            audio, sr = librosa.load(audio_path, 16000)
            array_path = audio_path.replace("wav", "npy")
            if not Path(array_path).is_file():
                np.save(open(array_path, 'wb'), audio)