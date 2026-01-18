# imports
import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from config import get_long_audio_dir, get_spectrogram_dir

# mel spectrogram parameters
number_mels = 256  # 256
hop_length = 64  # 64 for 100ms, 32 for 50ms, 16 for 25 ms

# setup paths
# save_path = "F:/spectrograms/New_SwallowSet_Test/"
# root_dir = "F:/datasets/New_SwallowSet/Test/"
root_dir = get_long_audio_dir()
save_path = get_spectrogram_dir()

if not os.path.exists(save_path):
    os.makedirs(save_path)

# list the class names
classes = os.listdir(root_dir)

for cl in classes:
    if not os.path.exists(save_path + cl + "/"):
        os.makedirs(save_path + cl + "/")

    specimens = os.listdir(root_dir + "/" + cl)

    for specimen in specimens:
        if not os.path.exists(save_path + cl + "/" + specimen + "/"):
            os.makedirs(save_path + cl + "/" + specimen + "/")

        files = os.listdir(root_dir + "/" + cl + "/" + specimen)

        for i, current_file in enumerate(files):
            # if i > 0:
            #     break

            # print progress
            print("Processing... class: " + cl + " - file: " + current_file)

            # read sample and convert to mono
            y, sr = librosa.load(
                root_dir + "/" + cl + "/" + specimen + "/" + current_file
            )
            S1 = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=number_mels, hop_length=hop_length
            )
            S1_log = librosa.power_to_db(S1, ref=np.max)

            # ensure all spectrograms are of the same size
            rows, cols = S1_log.shape
            if cols < 346:
                pad_width = 346 - cols
                S1_log = np.pad(
                    S1_log,
                    ((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=-80,
                )
            elif cols > 346:
                S1_log = S1_log[:, :346]

            file_name = os.path.splitext(current_file)[0]
            np.save(save_path + cl + "/" + specimen + "/" + file_name + ".npy", S1_log)

            # librosa.display.specshow(S1_log, sr=sr, x_axis='time', y_axis='mel')
            # plt.show()
