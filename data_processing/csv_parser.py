import pandas as pd
import os
import random  # Import random for selective dummy addition

def csv_to_yolo_annotations(csv_file, output_dir, image_width, image_height, box_size=30, add_dummy_prob=0.1):
    # Load the CSV file into a DataFrame, skipping the header row
    data = pd.read_csv(csv_file, names=["id", "x", "y", "frame"], skiprows=1)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each frame in the CSV
    for frame_num in range(1, 501):  # Assuming 500 frames per folder
        # Filter rows for the current frame
        frame_data = data[data["frame"] == frame_num]
        
        annotations = []
        
        if not frame_data.empty:
            # If there are particles in this frame, create particle annotations
            for _, row in frame_data.iterrows():
                particle_id, x, y = int(row["id"]), int(row["x"]), int(row["y"])
                
                # Calculate normalized YOLO values
                x_center = (x + box_size / 2) / image_width
                y_center = (y + box_size / 2) / image_height
                width = box_size / image_width
                height = box_size / image_height
                
                # Format as YOLO annotation (class_id x_center y_center width height)
                annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            # If there are no particles, add a dummy "background" annotation with some probability
            if random.random() < add_dummy_prob:
                # Use class_id "1" for background class
                annotations.append(f"1 0.1 0.1 0.2 0.2")

        # Save annotations to the corresponding .txt file
        frame_filename = f"frame_{frame_num:04d}.txt"  # Format: frame_0001.txt
        frame_filepath = os.path.join(output_dir, frame_filename)
        
        with open(frame_filepath, "w") as f:
            f.write("\n".join(annotations))
        print(f"Saved annotations for frame {frame_num} to {frame_filepath}")

# Paths to CSV files
Q1250_z35_mov_1_csv = r"C:\Users\sawye\Downloads\cvfinalprojdata\EV xslot data for CV\HCC1954REPX_Q1250_z35_mov01.csv"
Q1250_z35_mov_2_csv = r"C:\Users\sawye\Downloads\cvfinalprojdata\EV xslot data for CV\HCC1937_Q1250_z35_mov02.csv"
Q750_z40_mov_1_csv = r"C:\Users\sawye\Downloads\cvfinalprojdata\EV xslot data for CV\MDAMC231_Q750_z40_mov01.csv"

# Folder names for output
csvs = [Q1250_z35_mov_1_csv, Q1250_z35_mov_2_csv, Q750_z40_mov_1_csv]
folder_names = ["Q1250_z35_mov_1_annotations", "Q1250_z35_mov_2_annotations", "Q750_z40_mov_1_annotations"]

# Iterate through each video and process it
for i in range(3):
    csv_file = csvs[i]
    folder_name = folder_names[i]
    image_width = 1200
    image_height = 1200
    csv_to_yolo_annotations(csv_file, folder_name, image_width, image_height)
