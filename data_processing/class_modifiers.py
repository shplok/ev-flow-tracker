import os

def backgroundClassEditor(labels_folder):
    backgroundCount = 0
    particleCount = 0
    
    for filename in os.listdir(labels_folder):
        filepath = os.path.join(labels_folder, filename)
        
        # Open the file and read all lines
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Rewrite the lines, adjusting any background labels
        with open(filepath, 'w') as file:
            for line in lines:
                # Check if line matches the full-frame or small-corner background label
                if line.strip() == "1 0.5 0.5 1.0 1.0" or line.strip() == "1 0.05 0.05 0.05 0.05":
                    backgroundCount += 1
                    # Replace with smaller bounding box in the corner
                    new_background_label = "1 0.05 0.05 0.05 0.05\n"
                    file.write(new_background_label)
                    print(f"Updated Class Information in {filename}")
                else:
                    particleCount += 1
                    file.write(line)
                    print(f"No Changes Required in {filename}")
    
    # Print counts of particles and background frames
    print("Total Particles:", particleCount)
    print("Total Background Frames:", backgroundCount)
    

# Folder paths for labels
Q1250_z35_mov_1_csv = r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\labels\Q1250_z35_mov_1"
Q1250_z35_mov_2_csv = r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\labels\Q1250_z35_mov_2"
Q750_z40_mov_1_csv = r"C:\Users\sawye\Downloads\cs675finprojbucket\initialbucketforevflow\Final Project ooo\ev-flow-tracker\data\labels\Q750_z40_mov_1"

# List of label folders to process
labels_folders = [Q1250_z35_mov_1_csv, Q1250_z35_mov_2_csv, Q750_z40_mov_1_csv]

# Process each folder
for labels_folder in labels_folders:
    backgroundClassEditor(labels_folder)
