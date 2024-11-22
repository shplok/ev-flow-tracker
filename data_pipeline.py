import yaml
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import torch
import shutil
import os
from tqdm import tqdm

class ParticleTrainingPipeline:
    def __init__(self, data_dir, output_dir, img_size=1200):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.setup_directories()
        self.setup_augmentation()
        
    def setup_directories(self):
        """Create necessary directories for training"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create train and val directories
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        
    def setup_augmentation(self):
        """Setup augmentation pipeline"""
        self.transform = A.Compose([
            # Geometric Transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.5
            ),
            
            # Color/Intensity Transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.RandomGamma(p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.5),
            
            # Noise and Blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,  # Only keep boxes that are at least 30% visible
            check_each_transform=True  # Check bbox validity after each transform
    ))

    def prepare_data(self):
        """Prepare data from existing train/val structure with experiment subfolders"""
        print(f"\nFull data directory path: {self.data_dir.absolute()}")
        images_dir = self.data_dir / 'images'
        labels_dir = self.data_dir / 'labels'
        
        print("\nChecking directory structure...")
        
        # Get all experiment folders in train and val
        train_images_dir = images_dir / 'train'
        val_images_dir = images_dir / 'val'
        
        print("\nFound directories:")
        print(f"Train images base: {train_images_dir}")
        print(f"Val images base: {val_images_dir}")
        
        # Create output directories with absolute paths
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Process training data from all experiment folders
        print("\nProcessing training data...")
        all_train_images = []
        for exp_folder in train_images_dir.glob('*'):
            if exp_folder.is_dir():
                print(f"\nProcessing experiment folder: {exp_folder.name}")
                images = list(exp_folder.glob('*.jpg'))
                all_train_images.extend(images)
                print(f"Found {len(images)} images")
                
                # Copy images and labels
                for img_path in tqdm(images, desc=f"Copying {exp_folder.name}"):
                    # Copy image
                    shutil.copy2(img_path, train_dir / img_path.name)
                    # Copy corresponding label
                    label_path = labels_dir / 'train' / exp_folder.name / img_path.with_suffix('.txt').name
                    if label_path.exists():
                        shutil.copy2(label_path, train_dir / label_path.name)
                    else:
                        print(f"Warning: No label found for {img_path.name}")
        
        print(f"\nTotal training images processed: {len(all_train_images)}")
                
        # Process validation data from all experiment folders
        print("\nProcessing validation data...")
        all_val_images = []
        for exp_folder in val_images_dir.glob('*'):
            if exp_folder.is_dir():
                print(f"\nProcessing experiment folder: {exp_folder.name}")
                images = list(exp_folder.glob('*.jpg'))
                all_val_images.extend(images)
                print(f"Found {len(images)} images")
                
                # Copy images and labels
                for img_path in tqdm(images, desc=f"Copying {exp_folder.name}"):
                    # Copy image
                    shutil.copy2(img_path, val_dir / img_path.name)
                    # Copy corresponding label
                    label_path = labels_dir / 'val' / exp_folder.name / img_path.with_suffix('.txt').name
                    if label_path.exists():
                        shutil.copy2(label_path, val_dir / label_path.name)
                    else:
                        print(f"Warning: No label found for {img_path.name}")
        
        print(f"\nTotal validation images processed: {len(all_val_images)}")
        
        # Verify the data structure
        print("\nVerifying data structure...")
        print(f"Train directory exists: {train_dir.exists()}")
        print(f"Train directory contents: {len(list(train_dir.glob('*.jpg')))} images")
        print(f"Val directory exists: {val_dir.exists()}")
        print(f"Val directory contents: {len(list(val_dir.glob('*.jpg')))} images")

    def augment_dataset(self):
        """Augment training data"""
        train_dir = self.output_dir / 'train'
        aug_dir = self.output_dir / 'train_aug'
        aug_dir.mkdir(exist_ok=True)
        
        # Copy original data
        print("\nCopying original data...")
        for img_path in tqdm(list(train_dir.glob('*.jpg')), desc="Copying originals"):
            shutil.copy2(img_path, aug_dir / img_path.name)
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy2(label_path, aug_dir / label_path.name)
        
        # Augment data
        print("Creating augmented samples...")
        for img_path in tqdm(list(train_dir.glob('*.jpg')), desc="Augmenting"):
            self._augment_image(img_path, aug_dir)

    def _augment_image(self, img_path, output_dir):
        """Apply augmentation to a single image and its annotations"""
        # Read image and labels
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = img_path.with_suffix('.txt')
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(class_id)
        
        # Apply augmentation with error handling
        if bboxes:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Additional validation check
                valid_boxes = True
                for bbox in transformed['bboxes']:
                    x, y, w, h = bbox
                    # Check if any coordinates are outside valid range [0, 1]
                    if (x < 0 or y < 0 or 
                        x + w > 1 or y + h > 1 or 
                        w <= 0 or h <= 0):
                        valid_boxes = False
                        break
                
                # Only save if boxes are valid
                if valid_boxes and transformed['bboxes']:  # Also check if we have any boxes
                    # Save augmented image and labels
                    aug_img_path = output_dir / f'aug_{img_path.name}'
                    aug_label_path = aug_img_path.with_suffix('.txt')
                    
                    # Save image
                    cv2.imwrite(str(aug_img_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                    
                    # Save labels
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                            f.write(f'{int(class_id)} {" ".join(map(str, bbox))}\n')
                else:
                    print(f"Skipping augmented version of {img_path.name} due to invalid bounding boxes")
                    
            except ValueError as e:
                print(f"Skipping augmented version of {img_path.name} due to invalid transformation: {str(e)}")

    def create_yaml_config(self):
        """Create YAML configuration for YOLOv5"""
        # Use absolute paths
        config = {
            'train': r"C:\Users\sawye\OneDrive\Desktop\ev-flow-tracker\results\train_aug",  
            'val': r"C:\Users\sawye\OneDrive\Desktop\ev-flow-tracker\results\val",
            'nc': 1,  # number of classes
            'names': ['particle']  # class names
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return yaml_path

    def create_training_config(self):
        """Create training configuration"""
        config = {
            'model_config': {
                'cfg': 'yolov5s.yaml',
                'weights': 'yolov5s.pt',  # pre-trained weights
            },
            'training_params': {
                'epochs': 300,
                'batch_size': 16,
                'img_size': self.img_size,
                'patience': 50,  # early stopping patience
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'dropout': 0.2,
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1,
            }
        }
        
        config_path = self.output_dir / 'training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return config_path

    def train_model(self, yaml_config, training_config):
        """Train YOLOv5 model"""
        from yolov5 import train
        
        # Load training config
        with open(training_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Setup training arguments
        args = {
            'data': str(yaml_config),
            'cfg': config['model_config']['cfg'],
            'weights': config['model_config']['weights'],
            'epochs': config['training_params']['epochs'],
            'batch_size': config['training_params']['batch_size'],
            'img_size': config['training_params']['img_size'],
            'patience': config['training_params']['patience'],
            'project': str(self.output_dir / 'runs'),
            'name': 'train',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'Adam',
            'lr0': config['training_params']['lr0'],
            'lrf': config['training_params']['lrf'],
            'momentum': config['training_params']['momentum'],
            'weight_decay': config['training_params']['weight_decay'],
            'warmup_epochs': config['training_params']['warmup_epochs'],
            'warmup_momentum': config['training_params']['warmup_momentum'],
            'warmup_bias_lr': config['training_params']['warmup_bias_lr'],
            'dropout': config['training_params']['dropout'],
            'mosaic': config['training_params']['mosaic'],
            'mixup': config['training_params']['mixup'],
            'copy_paste': config['training_params']['copy_paste'],
        }
        
        # Train the model
        train.run(**args)

    def run_training(self):
        """Execute the complete training pipeline"""
        print("Starting training pipeline...")
        
        # Prepare data
        print("\nPreparing data...")
        self.prepare_data()
        
        # Verify output directories exist
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        
        if not val_dir.exists() or len(list(val_dir.glob('*.jpg'))) == 0:
            raise Exception("Validation directory is missing or empty")
        
        # Augment training data
        print("\nAugmenting training data...")
        self.augment_dataset()
        
        # Create training configuration
        print("\nCreating training configuration...")
        training_config = self.create_training_config()
        
        # Create YAML config
        print("Creating YAML configuration...")
        yaml_config = self.create_yaml_config()
        
        # Print final verification
        print("\nFinal verification before training:")
        print(f"YAML config path: {yaml_config}")
        print(f"Training config path: {training_config}")
        print(f"Train directory: {train_dir} (exists: {train_dir.exists()})")
        print(f"Val directory: {val_dir} (exists: {val_dir.exists()})")
        
        # Train model
        print("\nStarting training...")
        self.train_model(yaml_config, training_config)

if __name__ == '__main__':
    pipeline = ParticleTrainingPipeline(
        data_dir=r'C:\Users\sawye\OneDrive\Desktop\ev-flow-tracker\data',
        output_dir='results',
        img_size=1200
    )
    pipeline.run_training()