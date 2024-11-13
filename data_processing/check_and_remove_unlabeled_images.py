import os

def imagePurger(images_path, annotations_path):
    # Loop through all image files
    for image_file in os.listdir(images_path):
        if image_file.endswith('.png'):  # Adjust extension if necessary
            image_basename = os.path.splitext(image_file)[0]
            label_file = os.path.join(annotations_path, f"{image_basename}.txt")

            # Check if the corresponding label file exists
            if not os.path.exists(label_file):
                image_filepath = os.path.join(images_path, image_file)
                os.remove(image_filepath)
                print(f"Deleted image without label: {image_file}")

# Paths for the image folders
Q1250_z35_mov_1_images = r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\images\Q1250_z35_mov_1"
Q1250_z35_mov_2_images = r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\images\Q1250_z35_mov_2"
Q750_z40_mov_1_images = r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\images\Q750_z40_mov_1"

# Folder names for annotations
annotations_folders = [
    r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\labels\Q1250_z35_mov_1",
    r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\labels\Q1250_z35_mov_2",
    r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\labels\Q750_z40_mov_1"
    ]

for i in range(3):
    annotations_folder = annotations_folders[i]
    
    # Choose the correct images folder for each annotation folder
    if i == 0:
        images_folder = Q1250_z35_mov_1_images
    elif i == 1:
        images_folder = Q1250_z35_mov_2_images
    else:
        images_folder = Q750_z40_mov_1_images
    
    imagePurger(images_folder, annotations_folder)
