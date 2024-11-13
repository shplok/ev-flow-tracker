import os

# Paths to image directories for each video
image_dirs = [
    r"C:\Users\Sawyer\Desktop\Important Docs\Umass Boston\FALL 2024\CS675\Final Project ooo\data\images\Q1250_z35_mov_1",
    r"C:\Users\Sawyer\Desktop\Important Docs\Umass Boston\FALL 2024\CS675\Final Project ooo\data\images\Q1250_z35_mov_2",
    r"C:\Users\Sawyer\Desktop\Important Docs\Umass Boston\FALL 2024\CS675\Final Project ooo\data\images\Q750_z40_mov_1"
]

# Paths to corresponding label directories for each video
labels_dirs = [
    r"C:\Users\Sawyer\Desktop\Important Docs\Umass Boston\FALL 2024\CS675\Final Project ooo\data\labels\Q1250_z35_mov_1",
    r"C:\Users\Sawyer\Desktop\Important Docs\Umass Boston\FALL 2024\CS675\Final Project ooo\data\labels\Q1250_z35_mov_2",
    r"C:\Users\Sawyer\Desktop\Important Docs\Umass Boston\FALL 2024\CS675\Final Project ooo\data\labels\Q750_z40_mov_1"
]

# Loop through each video folder
for i in range(3):
    images_dir = image_dirs[i]
    labels_dir = labels_dirs[i]

    # Ensure labels directory exists
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Get list of all image files in the current video folder
    image_files = sorted(os.listdir(images_dir))

    # Loop over the range of the number of images
    for image_file in image_files:
        # Construct the corresponding label file name (same base name)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if annotation exists, and if not, create an empty annotation file
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                # Add dummy data: class 0, centered bounding box (normalized)
                f.write("0 0.5 0.5 0.1 0.1\n")  # Dummy annotation for an empty frame
            print(f"Created dummy annotation: {label_path}")
