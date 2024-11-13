import pandas as pd
import os
import cv2
import tifffile

def generate_bounding_boxes(frames_dir, csv_file, folder_name, box_size=52, output_dir="frames_with_bboxes", annotations_file="annotations.txt"):
    # Load particle CSV data
    particle_data = pd.read_csv(csv_file, names=["id", "x", "y", "frame"])
    
    # Ensure 'frame' column is numeric and drop invalid rows
    particle_data["frame"] = pd.to_numeric(particle_data["frame"], errors="coerce")
    particle_data.dropna(subset=["frame"], inplace=True)
    particle_data["frame"] = particle_data["frame"].astype(int)  # Convert 'frame' to integer
    
    # Create output subfolder for this video using the provided folder name
    video_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Open the annotations file to save bounding box information
    with open(annotations_file, "w") as f:
        # Open the TIFF file using tifffile
        with tifffile.TiffFile(frames_dir) as tif:
            num_frames = len(tif.pages)  # Number of pages (frames) in the TIFF file
            print(f"Total frames in TIFF: {num_frames}")
            
            # Iterate through each frame in the TIFF
            for frame_num in range(num_frames):
                frame = tif.pages[frame_num].asarray()  # Read the frame data as numpy array
                
                # Draw bounding boxes and save annotations
                particles_in_frame = particle_data[particle_data['frame'] == frame_num + 1]
                for _, row in particles_in_frame.iterrows():
                    particle_id, x, y = row["id"], int(row["x"]), int(row["y"])
                    
                    # Define the bounding box coordinates
                    x1, y1 = x - box_size // 2, y - box_size // 2
                    x2, y2 = x + box_size // 2, y + box_size // 2
                    
                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(particle_id), (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Write annotation in YOLO format: <class_id> <x_center> <y_center> <width> <height>
                    x_center = (x1 + x2) / 2 / frame.shape[1]
                    y_center = (y1 + y2) / 2 / frame.shape[0]
                    width = box_size / frame.shape[1]
                    height = box_size / frame.shape[0]
                    
                    f.write(f"{x_center} {y_center} {width} {height}\n")
                
                # Save the frame with drawn bounding boxes as a .png or .tif
                output_frame_path = os.path.join(video_output_dir, f"frame_{frame_num + 1:04d}.png")  # Output as .png or .tif
                cv2.imwrite(output_frame_path, frame)
                print(f"Processed frame {frame_num + 1} with bounding boxes")


# Example paths
Q1250_z35_mov_1_csv = r"C:\Users\Sawyer\Downloads\OneDrive_2024-11-07\EV xslot data for CV\HCC1954REPX_Q1250_z35_mov01.csv"
Q1250_z35_mov_1_frame_dir = r"C:\Users\Sawyer\Downloads\OneDrive_2024-11-07\EV xslot data for CV\xslot_HCC1954REPX_01_1250uLhr_z35um_mov_1_MMStack_Pos0.ome.tif"

Q1250_z35_mov_2_csv = r"C:\Users\Sawyer\Downloads\OneDrive_2024-11-07\EV xslot data for CV\HCC1937_Q1250_z35_mov02.csv"
Q1250_z35_mov_2_frame_dir = r"C:\Users\Sawyer\Downloads\OneDrive_2024-11-07\EV xslot data for CV\xslot_HCC1937_expt01_1250uLhr_z35um_mov_2_MMStack_Pos0.ome.tif"

Q750_z40_mov_1_csv = r"C:\Users\Sawyer\Downloads\OneDrive_2024-11-07\EV xslot data for CV\MDAMC231_Q750_z40_mov01.csv"
Q750_z40_mov_1_frame_dir = r"C:\Users\Sawyer\Downloads\OneDrive_2024-11-07\EV xslot data for CV\xslot_MDAMC231_750uLhr_z40um_mov_1_MMStack_Pos0.ome.tif"

# Define the folder names you want to use for each dataset
folder_names = [
    "Q1250_z35_mov_1",
    "Q1250_z35_mov_2",
    "Q750_z40_mov_1"
]

csvs = [Q1250_z35_mov_1_csv, Q1250_z35_mov_2_csv, Q750_z40_mov_1_csv]
frame_dirs = [Q1250_z35_mov_1_frame_dir, Q1250_z35_mov_2_frame_dir, Q750_z40_mov_1_frame_dir]

# Iterate through each video and process it
for i in range(3):
    frames_dir = frame_dirs[i]
    csv_file = csvs[i]
    folder_name = folder_names[i]
    generate_bounding_boxes(frames_dir, csv_file, folder_name)
