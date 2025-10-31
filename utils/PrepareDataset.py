"""
This script prepares the SwallowSet dataset by organizing audio files into a structured directory format.
It reads audio files from the raw dataset, processes them, and saves them into a new directory, ensuring correct sample rate and channel configuration for each specimen's file and grouping them together.
Needs to be run only once for data cleanup and organization.
"""

import os
import librosa
import soundfile
from config import get_raw_data_dir, get_short_audio_dir

# init
count = 0
root_dir = get_raw_data_dir()
save_path = get_short_audio_dir()
if not os.path.exists(save_path):
    os.makedirs(save_path)
classes = os.listdir(root_dir)

# iterate through the dataset
for cl in classes:

    if not os.path.exists( os.path.join(save_path, cl)):
        os.makedirs(os.path.join(save_path, cl))

    specimens = os.listdir(os.path.join(root_dir, cl))

    for specimen in specimens:

        specimen_id = specimen.split("_")[0]

        specimen_folder = os.path.join(save_path, cl, specimen_id)
        if not os.path.exists(specimen_folder):
            os.makedirs(specimen_folder)

        subfolder = os.listdir(os.path.join(root_dir, cl, specimen))
        # subfolder = [f for f in subfolder if not ".csv" in f]
        subfolder = [f for f in subfolder if not ".csv" in f and "1sec" in f]

        files = os.listdir(os.path.join(root_dir, cl, specimen, subfolder[0]))
        wav_files = [f for f in files if (".wav" in f or ".WAV" in f)]

        for file in wav_files:

            print("Processing file: " + file)

            # split file extension
            filename, file_extension = os.path.splitext(file)

            # read sample
            wav2, sr = librosa.load(os.path.join(root_dir, cl, specimen, subfolder[0], file),
                                    sr=None, mono=True)

            # copy original sample
            save_filename = filename + ".wav"
            soundfile.write(os.path.join(specimen_folder, save_filename), wav2, sr)

            count += 1

print("Total files processed: " + str(count))

