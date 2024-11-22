from pathlib import Path
import subprocess
import sys
import os

def run_detection(
    weights_path='results/runs/train/weights/best.pt',
    source=r'C:\Users\sawye\OneDrive\Desktop\ev-flow-tracker\detectset', 
    output_dir='results/detections',
    conf_thres=0.25,
    iou_thres=0.45,
    img_size=1216,
):
    """Run detection using trained model via command line"""
    # Get absolute paths
    weights_path = str(Path(weights_path).resolve())
    source = str(Path(source).resolve())
    output_path = Path(output_dir).resolve()
    yolov5_path = Path(__file__).parent / 'yolov5'
    detect_script = yolov5_path / 'detect.py'
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Construct command
    cmd = [
        sys.executable,  # Python interpreter path
        str(detect_script),
        '--weights', weights_path,
        '--source', source,
        '--project', str(output_path),
        '--name', 'exp',
        '--conf-thres', str(conf_thres),
        '--iou-thres', str(iou_thres),
        '--imgsz', str(img_size),
        '--save-txt',
        '--save-conf',
        '--exist-ok'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the detection script
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running detection: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == '__main__':
    try:
        run_detection(
            weights_path='results/runs/train/weights/best.pt',
            source=r'C:\Users\sawye\OneDrive\Desktop\ev-flow-tracker\detectset',
            output_dir='results/detections',
            conf_thres=0.25,
            img_size=1216
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")