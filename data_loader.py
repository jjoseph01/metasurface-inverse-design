# data_loader.py
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from PIL import Image
import os
import numpy as np

from config import (
    CHANNELS, NUM_ANGLES, SIMULATOR_IMAGE_SIZE, TEST_SPLIT_SIZE,
    BATCH_SIZE, WORKERS, CURRENT_GAN_IMAGE_SIZE
)

class DiverseMetasurfaceDataset(Dataset):
    """
    Custom Dataset from simulator4.ipynb, adapted for the project.
    It loads metasurface images and their absorbance spectra from a single
    metadata file and a root image folder.
    """
    def __init__(self, metadata_file, image_folder_path, transform=None):
        try:
            self.metadata_df = pd.read_csv(metadata_file)
        except FileNotFoundError:
            print(f"Error: Metadata file not found at {metadata_file}")
            raise
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.absorbance_cols = [col for col in self.metadata_df.columns if col.startswith('Absorbance')]
        if len(self.absorbance_cols) != NUM_ANGLES:
            raise ValueError(f"Found {len(self.absorbance_cols)} absorbance columns, but expected {NUM_ANGLES}")

        self.filename_col = self.metadata_df.columns[0]
        self.metadata_df['class'] = self.metadata_df[self.filename_col].apply(lambda x: os.path.split(os.path.dirname(x))[-1] if os.path.dirname(x) else 'unknown')

        # --- CORRECTED LOGIC ---
        # Build a map of the base filename to its full, absolute path.
        self.image_path_map = {
            f: os.path.join(dp, f)
            for dp, dn, fn in os.walk(self.image_folder_path)
            for f in fn if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        }

        # Filter the dataframe to only include rows where the image file exists in our map.
        # This handles cases where the CSV has 'image.png' or 'folder/image.png'.
        original_len = len(self.metadata_df)
        self.metadata_df = self.metadata_df[self.metadata_df[self.filename_col].apply(lambda x: os.path.basename(x) in self.image_path_map)]
        
        if len(self.metadata_df) < original_len:
            print(f"Warning: Dropped {original_len - len(self.metadata_df)} rows from CSV due to missing image files.")
        if len(self.metadata_df) == 0:
            raise ValueError(f"No matching image files found for entries in {metadata_file} within {self.image_folder_path} (recursively).")

    def __len__(self):
        return len(self.metadata_df)

    def get_labels(self):
        return self.metadata_df['class'].tolist()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row_data = self.metadata_df.iloc[idx]
        # Get the filename from the CSV, which might be 'wedge_0015.png' or 'folder/wedge_0015.png'
        img_filename_in_csv = row_data[self.filename_col]
        # Use only the base filename as the key for our map
        base_filename = os.path.basename(img_filename_in_csv)
        
        # Get the full, absolute path from the map
        img_full_path = self.image_path_map.get(base_filename)

        if not img_full_path:
             # This should not happen due to the filtering in __init__
             raise FileNotFoundError(f"Image file '{base_filename}' not found in image map during __getitem__.")

        image = Image.open(img_full_path).convert('L')
        absorbance_vector = row_data[self.absorbance_cols].values.astype(np.float32)
        absorbance_tensor = torch.from_numpy(absorbance_vector)

        if self.transform:
            image = self.transform(image)
        return image, absorbance_tensor

class ImageOnlyDataset(Dataset):
    def __init__(self, image_folder_path, transform=None):
        self.image_folder_path = image_folder_path
        self.image_files = self._find_image_files_recursive(image_folder_path)
        self.transform = transform

    def _find_image_files_recursive(self, folder_path):
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_full_path = self.image_files[idx]
        image = Image.open(img_full_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

def get_dataloaders(mode, image_folder_path, metadata_file):
    """
    Sets up and returns dataloaders by loading a single metadata file and
    splitting it into training and testing sets.
    """
    gan_transforms = transforms.Compose([
        transforms.Resize(CURRENT_GAN_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(CURRENT_GAN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS)
    ])
    sim_train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize(SIMULATOR_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIMULATOR_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS)
    ])
    sim_test_transforms = transforms.Compose([
        transforms.Resize(SIMULATOR_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIMULATOR_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS)
    ])

    try:
        # Create the full dataset from the single compiled metadata file
        full_dataset_for_split = DiverseMetasurfaceDataset(
            metadata_file=metadata_file,
            image_folder_path=image_folder_path
        )
        labels = full_dataset_for_split.get_labels()

        # Perform the stratified split to get train and test indices
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT_SIZE, random_state=42)
        train_indices, test_indices = next(sss.split(np.zeros(len(labels)), labels))

        # Create two dataset objects with different transforms
        train_dataset_sim_obj = DiverseMetasurfaceDataset(metadata_file=metadata_file, image_folder_path=image_folder_path, transform=sim_train_transforms)
        test_dataset_sim_obj = DiverseMetasurfaceDataset(metadata_file=metadata_file, image_folder_path=image_folder_path, transform=sim_test_transforms)

        # Create PyTorch Subsets using the indices
        train_subset_sim = Subset(train_dataset_sim_obj, train_indices)
        test_subset_sim = Subset(test_dataset_sim_obj, test_indices)

        sim_train_loader = DataLoader(train_subset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
        sim_test_loader = DataLoader(test_subset_sim, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

        print(f"Simulator datasets created from '{os.path.basename(metadata_file)}'. Total: {len(full_dataset_for_split)}, Train: {len(train_subset_sim)}, Test: {len(test_subset_sim)}")

        if mode == 'CONSTANT_TARGET':
            gan_unconditional_dataset = ImageOnlyDataset(image_folder_path=image_folder_path, transform=gan_transforms)
            gan_loader = DataLoader(gan_unconditional_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
            print(f"GAN unconditional dataset loaded with {len(gan_unconditional_dataset)} samples.")
            return sim_train_loader, sim_test_loader, gan_loader, full_dataset_for_split

        elif mode == 'CONDITIONAL':
            gan_train_dataset = DiverseMetasurfaceDataset(metadata_file=metadata_file, image_folder_path=image_folder_path, transform=gan_transforms)
            gan_loader = DataLoader(gan_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
            print(f"GAN conditional dataset loaded with {len(gan_train_dataset)} samples.")
            return sim_train_loader, sim_test_loader, gan_loader, gan_train_dataset

        else:
            raise ValueError(f"Unknown training mode: {mode}")

    except Exception as e:
        print(f"Error setting up dataset/dataloader: {e}")
        raise

def resize_for_simulator(images):
    """Resizes images to the simulator's required input size."""
    resizer = transforms.Resize(SIMULATOR_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC)
    return resizer(images)
