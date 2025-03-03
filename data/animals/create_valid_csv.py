import os
import csv

main_classes_only = True  # customize this!

folder_contents = os.listdir("./")
main_class_dirs = [f for f in folder_contents
                   if os.path.isdir(os.path.join("./", f)) and not f.endswith(".")]
main_class_dirs = sorted(main_class_dirs)  # sorting alphabetically
main_class_count = len(main_class_dirs)

classes = ["albatross", "cheetah", "giraffe", "ostrich", "penguin", "tiger", "zebra",
           "bird", "black", "blackstripes", "blackwhite", "carnivore", "claws", "cud", "darkspots", "eventoed",
           "feather", "fly", "forwardeyes", "goodflier", "hair", "hoofs", "layeggs", "longlegs", "longneck", "mammal",
           "meat", "milk", "pointedteeth", "swim", "tawny", "ungulate", "white"]

targets = {
    "albatross": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cheetah": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    "giraffe": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    "ostrich": [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "penguin": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "tiger": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    "zebra": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1]
}

files = []
labels = []

for main_class_name in main_class_dirs:
    class_folder = os.path.join(main_class_name)
    folder_contents = os.listdir(class_folder)
    new_files = [os.path.join(class_folder, f) for f in folder_contents
                 if os.path.isfile(os.path.join(class_folder, f)) and f.endswith(".jpg")]
    new_files.sort()
    binary_target = targets[main_class_name]

    for i, new_file in enumerate(new_files):
        new_labels = []
        for j, bit in enumerate(binary_target):
            if main_classes_only and j >= main_class_count:
                continue
            if bit == 1:
                new_labels.append(classes[j])
        files.append(new_file)
        labels.append(new_labels)

csv_filename = 'labels.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    for i in range(len(files)):
        row = [files[i]] + labels[i]  # combine file path with the corresponding labels
        writer.writerow(row)
