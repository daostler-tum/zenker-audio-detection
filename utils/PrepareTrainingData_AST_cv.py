import os
import random
import numpy as np
import json
from collections import defaultdict
from config import get_short_audio_dir

dataset_root = get_short_audio_dir()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "../data_ast_cv/")
seed = 42
random.seed(seed)

# Centralized class to index mapping
CLASS_TO_INDEX = {
    "Idle": 0,
    "Healthy": 1,
    "Zenker": 2,
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#save mapping for reference
with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
    json.dump(CLASS_TO_INDEX, f, indent=2)

# Group specimens by class
class_specimens = defaultdict(list)
classes = os.listdir(dataset_root)
for cl in classes:
    specimens = os.listdir(os.path.join(dataset_root, cl))
    class_specimens[cl] = specimens


patho_classes = ["Healthy", "Zenker"]
# create a list of all all pathology subjects with their lables
patho_subjects = []
for p in patho_classes:
    patho_subjects.extend([(s, p) for s in class_specimens[p]])

# 5-fold stratified split at pathology subject level
from sklearn.model_selection import StratifiedKFold
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
patho_subjects = np.array(patho_subjects)
splits = list(skf.split(patho_subjects, [label for _, label in patho_subjects]))

subject_folds = []  # list of (train_pathology_subjects, test_pathology_subjects)
for train_idx, test_idx in splits:
    test_specimens = patho_subjects[test_idx, 0].tolist()  # just the specimen names
    train_specimens = patho_subjects[train_idx, 0].tolist()
    subject_folds.append((set(train_specimens), set(test_specimens)))



# class_specimens_patho = {cl: class_specimens[cl] for cl in patho_classes}

# # 5-fold split for each class (healthy/zenker only)
# from sklearn.model_selection import KFold
# num_folds = 5
# folds = {cl: [] for cl in patho_classes}
# for cl in patho_classes:
#     specimens = class_specimens_patho[cl]
#     kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
#     specimens = np.array(specimens)
#     for _, test_idx in kf.split(specimens):
#         folds[cl].append(specimens[test_idx].tolist())


# Use CLASS_TO_INDEX mapping to get labels from folder names
def get_data_labels(folders):
    x, y = [], []
    for folder in folders:
        class_name = os.path.basename(os.path.dirname(folder.rstrip('/')))
        if class_name not in CLASS_TO_INDEX:
            raise ValueError(f"Unknown class folder: {class_name} (from {folder})")
        label = CLASS_TO_INDEX[class_name]
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                x.append(os.path.join(folder, file))
                y.append(label)
    return x, y


for fold_idx, (train_subjects, test_subjects) in enumerate(subject_folds, start=1):
    # Build folder lists from pathology subjects
    train_folders = []
    test_folders = []
    train_specimen_ids = []  # entries like Class/specimen
    test_specimen_ids = []

    # Pathology (Healthy + Zenker)
    for cls in patho_classes:
        # training pathology specimens of this class
        for specimen in sorted(s for s in class_specimens[cls] if s in train_subjects):
            train_folders.append(os.path.join(dataset_root, cls, specimen, ""))
            train_specimen_ids.append(f"{cls}/{specimen}")
        # test pathology specimens
        for specimen in sorted(s for s in class_specimens[cls] if s in test_subjects):
            test_folders.append(os.path.join(dataset_root, cls, specimen, ""))
            test_specimen_ids.append(f"{cls}/{specimen}")

    # Idle attachment (only for subjects that appear in pathology sets)
    if "Idle" in class_specimens:
        for specimen in sorted(class_specimens["Idle"]):
            if specimen in train_subjects:
                train_folders.append(os.path.join(dataset_root, "Idle", specimen, ""))
                train_specimen_ids.append(f"Idle/{specimen}")
            elif specimen in test_subjects:
                test_folders.append(os.path.join(dataset_root, "Idle", specimen, ""))
                test_specimen_ids.append(f"Idle/{specimen}")

    # Derive file paths and labels
    train_x, train_y = get_data_labels(train_folders)
    test_x, test_y = get_data_labels(test_folders)

    print(f"\nFold {fold_idx}")
    print("Training data length:", len(train_x))
    print("Test data length:", len(test_x))

    # Persist arrays
    np.save(f"{output_dir}train_x_fold{fold_idx}.npy", train_x)
    np.save(f"{output_dir}train_y_fold{fold_idx}.npy", train_y)
    np.save(f"{output_dir}test_x_fold{fold_idx}.npy", test_x)
    np.save(f"{output_dir}test_y_fold{fold_idx}.npy", test_y)

    # CSV labels
    with open(f"{output_dir}train_labels_fold{fold_idx}.csv", "w") as f:
        f.write("file,label\n")
        for file, label in zip(train_x, train_y):
            f.write(f"{file},{label}\n")
    with open(f"{output_dir}test_labels_fold{fold_idx}.csv", "w") as f:
        f.write("file,label\n")
        for file, label in zip(test_x, test_y):
            f.write(f"{file},{label}\n")

    # Specimen IDs
    with open(f"{output_dir}train_ids_fold{fold_idx}.txt", "w") as f:
        for item in train_specimen_ids:
            f.write(item + "\n")
    with open(f"{output_dir}test_ids_fold{fold_idx}.txt", "w") as f:
        for item in test_specimen_ids:
            f.write(item + "\n")

    # Build and save metadata JSON
    train_subjects_pathology = sorted([s for s in train_subjects])
    test_subjects_pathology = sorted([s for s in test_subjects])

    # Class distribution (file-level)
    def distro(arr):
        import numpy as np
        u,c = np.unique(arr, return_counts=True)
        return {INDEX_TO_CLASS[int(k)]: int(v) for k,v in zip(u,c)}

    meta = {
        "fold": fold_idx,
        "train_pathology_subjects": train_subjects_pathology,
        "test_pathology_subjects": test_subjects_pathology,
        "num_train_pathology_subjects": len(train_subjects_pathology),
        "num_test_pathology_subjects": len(test_subjects_pathology),
        "idle_in_train_subjects": sorted([s for s in train_subjects_pathology if s in class_specimens.get("Idle", [])]),
        "idle_in_test_subjects": sorted([s for s in test_subjects_pathology if s in class_specimens.get("Idle", [])]),
        "num_train_files": len(train_x),
        "num_test_files": len(test_x),
        "file_class_distribution_train": distro(np.array(train_y, dtype=int)),
        "file_class_distribution_test": distro(np.array(test_y, dtype=int)),
        "labels_mapping": CLASS_TO_INDEX,
    }
    with open(os.path.join(output_dir, f"fold{fold_idx}_meta.json"), "w") as mf:
        json.dump(meta, mf, indent=2)

for i, (tr, te) in enumerate(subject_folds, start=1):
    healthy_tr = sum(1 for s in tr if s in class_specimens["Healthy"])
    zenker_tr  = sum(1 for s in tr if s in class_specimens["Zenker"])
    healthy_te = sum(1 for s in te if s in class_specimens["Healthy"])
    zenker_te  = sum(1 for s in te if s in class_specimens["Zenker"])
    print(f"Fold {i}: Train(H={healthy_tr}, Z={zenker_tr})  Test(H={healthy_te}, Z={zenker_te})")

# just some check, that we did not miss any idle specimens
unmatched_idle = [s for s in class_specimens["Idle"] 
                  if s not in class_specimens["Healthy"] and s not in class_specimens["Zenker"]]
if unmatched_idle:
    print(f"Ignored {len(unmatched_idle)} idle specimens with no pathology match: {unmatched_idle[:5]}{'...' if len(unmatched_idle)>5 else ''}")
