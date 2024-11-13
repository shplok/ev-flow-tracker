import os

def annotationPurger(annotations_path, images_path):
    # Loop through all annotation files
    for annotation_file in os.listdir(annotations_path):
        annotation_filepath = os.path.join(annotations_path, annotation_file)
    
        # Open and check if the annotation file contains any object labels
        with open(annotation_filepath, 'r') as f:
            lines = f.readlines()
        
        # Check if annotation is empty or only contains the background label
        if len(lines) == 0 or all(line.startswith('background_class_id') for line in lines):
            # Get corresponding image filename
            image_file = annotation_file.replace('.txt', '.png')  # Adjust extension if needed
            image_filepath = os.path.join(images_path, image_file)
        
            # Delete annotation file and image file
            os.remove(annotation_filepath)
            if os.path.exists(image_filepath):
                os.remove(image_filepath)
            print(f"Removed empty annotation and image: {annotation_file} and {image_file}")

Q1250_z35_mov_1_images = r"C:\Users\sawye\Downloads\cvfinalprojdata\EV xslot data for CV\HCC1954REPX_Q1250_z35_mov01_images"
Q1250_z35_mov_2_images = r"C:\Users\sawye\Downloads\cvfinalprojdata\EV xslot data for CV\HCC1937_Q1250_z35_mov02_images"
Q750_z40_mov_1_images = r"C:\Users\sawye\Downloads\cvfinalprojdata\EV xslot data for CV\MDAMC231_Q750_z40_mov01_images"

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
    
    annotationPurger(annotations_folder, images_folder)
