from torch.utils.data import Dataset

import os

import random

import torch

import numpy as np

from PIL import Image

import cv2

from diffusers.utils import load_image


CROP_x= 512
CROP_y= 320


NUM_BINS= 6

EV_CHANNELS= NUM_BINS



'''
For 2x 
'''

UPSAMPLE_SCALE= [2]


def get_views(panorama_height, panorama_width, overlap_ratio=0.5):
    
    panorama_height /= 8
    panorama_width /= 8

    print('panorama_height', panorama_height)
    print('panorama_width', panorama_width)

    window_size_x= CROP_x // 8
    window_size_y= CROP_y // 8


    stride_x = int(window_size_x * (1 - overlap_ratio))
    stride_y = int(window_size_y * (1 - overlap_ratio))


    # num_blocks_height = (panorama_height - window_size_y) // stride_y + 1
    # num_blocks_width = (panorama_width - window_size_x) // stride_x + 1

    # account for residual blocks
    num_blocks_height = (panorama_height - window_size_y) // stride_y + 1
    if (panorama_height - window_size_y) % stride_y != 0:
        num_blocks_height += 1 
    
    num_blocks_width = (panorama_width - window_size_x) // stride_x + 1
    if (panorama_width - window_size_x) % stride_x != 0:
        num_blocks_width += 1



    total_num_blocks = int(num_blocks_height * num_blocks_width)

    views = []

    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride_y)
        h_end = h_start + window_size_y
        if h_end > panorama_height:
            h_end = panorama_height
            h_start = h_end - window_size_y
        w_start = int((i % num_blocks_width) * stride_x)
        w_end = w_start + window_size_x
        if w_end > panorama_width:
            w_end = panorama_width
            w_start = w_end - window_size_x
        
        views.append((w_start, h_start, w_end, h_end))


    return views


def apply_crop(image, start_x, start_y, crop_size_x, crop_size_y):
    image_croped = image.crop((start_x, start_y, start_x + crop_size_x, start_y + crop_size_y))

    return image_croped  


def get_random_crop_idx(image, crop_size_x, crop_size_y):
    width, height = image.size

    start_x = random.randint(0, width - crop_size_x)
    start_y = random.randint(0, height - crop_size_y)

    return start_x, start_y

def divide_image(image, crop_size_x, crop_size_y):
    width, height = image.size

    num_parts_x = width // crop_size_x
    num_parts_y = height // crop_size_y
    
    residual_x = width % crop_size_x
    residual_y = height % crop_size_y

    image_parts = []

    for i in range(num_parts_x):
        for j in range(num_parts_y):
            start_x = i * crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0:
        for j in range(num_parts_y):
            start_x = width - crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_y > 0:
        for i in range(num_parts_x):
            start_x = i * crop_size_x
            start_y = height - crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0 and residual_y > 0:
        start_x = width - crop_size_x
        start_y = height - crop_size_y

        image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

        image_parts.append(image_croped)


    return image_parts


def get_image_parts(image, crop_size_x, crop_size_y, idx):
    width, height = image.size

    num_parts_x = width // crop_size_x
    num_parts_y = height // crop_size_y
    
    residual_x = width % crop_size_x
    residual_y = height % crop_size_y

    image_parts = []

    for i in range(num_parts_x):
        for j in range(num_parts_y):
            start_x = i * crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0:
        for j in range(num_parts_y):
            start_x = width - crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_y > 0:
        for i in range(num_parts_x):
            start_x = i * crop_size_x
            start_y = height - crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0 and residual_y > 0:
        start_x = width - crop_size_x
        start_y = height - crop_size_y

        image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

        image_parts.append(image_croped)


    return image_parts[idx]

def check_all_files_exist(file_list, folder_path):

    return all([os.path.isfile(os.path.join(folder_path, file)) for file in file_list])

ALLOWED_FOLDERS = [
    "Eadom_1", "Eadom_2", "Eadom_3", "Eadom_4", "Eadom_5",
    "Merawi_1", "Merawi_2", "Merawi_3",
    "Sirguta_1", "Sirguta_2", "Sirguta_3",
]

class DummyDataset(Dataset):
    def __init__(self, base_folder, samples_per_folder=100,
                 width=512, height=320, sample_frames=14):
        self.base_folder = base_folder
        self.folders = [
            f for f in ALLOWED_FOLDERS
            if os.path.isdir(os.path.join(base_folder, f))
        ]
        if not self.folders:
            raise ValueError(f"No valid folders found under {base_folder}")

        self.samples_per_folder = samples_per_folder
        self.channels = 3
        self.ev_channels = EV_CHANNELS  # kept for shape compat only
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

        # Pre-sort frame lists once at init — avoids repeated os.listdir calls
        # Filters to .jpg only so no hidden files or metadata files sneak in
        self.folder_frames = {}
        for folder_name in self.folders:
            frames_path = os.path.join(base_folder, folder_name, 'frames')
            all_frames = sorted([
                f for f in os.listdir(frames_path)
                if f.lower().endswith('.jpg')
            ])
            if len(all_frames) < self.sample_frames:
                raise ValueError(
                    f"'{folder_name}/frames' has only {len(all_frames)} .jpg frames, "
                    f"need at least {self.sample_frames}."
                )
            self.folder_frames[folder_name] = all_frames

    def __len__(self):
        return len(self.folders) * self.samples_per_folder

    def __getitem__(self, idx):
        # Round-robin across folders so all folders are covered each epoch
        folder_name = self.folders[idx % len(self.folders)]
        frames_path = os.path.join(self.base_folder, folder_name, 'frames')
        frames = self.folder_frames[folder_name]

        # Random contiguous clip (e.g. frame_008281.jpg ... frame_008285.jpg)
        start_idx = random.randint(0, len(frames) - self.sample_frames)
        selected_frames = frames[start_idx:start_idx + self.sample_frames]

        scale = UPSAMPLE_SCALE[0]

        # Get crop anchor from first frame of clip
        first_img = Image.open(os.path.join(frames_path, selected_frames[0]))
        rand_x, rand_y = get_random_crop_idx(first_img, CROP_x, CROP_y)
        first_img.close()

        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width)
        )

        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(frames_path, frame_name)
            if i == 0:
                init_frame_path = frame_path

            with Image.open(frame_path) as img:
                w, h = img.size
                img = img.resize((int(w * scale), int(h * scale)))
                img_cropped = apply_crop(img, rand_x, rand_y, CROP_x, CROP_y)
                img_tensor = torch.from_numpy(np.array(img_cropped)).float()
                img_normalized = img_tensor / 127.5 - 1
                pixel_values[i] = img_normalized.permute(2, 0, 1)  # HWC -> CHW

        # All-zero event tensor — correct shape for torch.zeros_like() in train.py
        # No event folder is read; train.py already discards the values anyway
        ev_pixel_values = torch.zeros(
            (self.sample_frames, self.ev_channels, self.height, self.width),
            dtype=torch.float32
        )

        return {
            'pixel_values': pixel_values,
            'event_values': ev_pixel_values,
            'init_frame': init_frame_path,
        }
    

def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return img.crop((left, top, right, bottom))


def get_valid_image_bins(image_path, idx, width, height, num_frames, scale=UPSAMPLE_SCALE[0]):
    """
    Loads a validation clip from a frames/ folder (jpg files, no events).
    image_path: full path to the frames/ folder (e.g. .../Eadom_1/frames)
    idx: starting frame index
    """
    frames = sorted([f for f in os.listdir(image_path) if f.lower().endswith('.jpg')])

    if len(frames) < num_frames:
        raise ValueError(f"Validation folder '{image_path}' has only {len(frames)} frames, need {num_frames}.")

    # Clamp idx so we never go out of bounds
    idx = min(idx, len(frames) - num_frames)
    selected_frames = frames[idx:idx + num_frames]

    # Load first frame as the conditioning image
    first_frame_path = os.path.join(image_path, selected_frames[0])
    valid_image = load_image(first_frame_path)

    w, h = valid_image.size
    valid_image = valid_image.resize((int(w * scale), int(h * scale)))
    valid_image = center_crop(valid_image, width, height)

    # All-zero event tensor — events disabled
    ev_pixel_values = torch.zeros((num_frames, EV_CHANNELS, height, width), dtype=torch.float32)

    return valid_image, ev_pixel_values


class ValidDataset(Dataset):
    def __init__(self, base_folder, num_samples=100000, width=1024, height=576, sample_frames=14):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        # Define the path to the folder containing video frames
        # self.base_folder =  '/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/Control_SVD/Finetune_data/bdd100k/images/track/mini'

        # self.base_folder =  '/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/Control_SVD/Finetune_data/ev_svd_mini_rgb'
        self.base_folder = base_folder
        self.folders = sorted(os.listdir(self.base_folder))
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx= 0):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
   

        folder_idx= idx[0]
        frame_idx= idx[1]
        nparts= idx[2]

        skip= idx[3]

        # slecet a folder by index
        chosen_folder = self.folders[folder_idx]
        folder_path = os.path.join(self.base_folder, chosen_folder)


        # Get from rgb folder
        rgb_folder_path = os.path.join(folder_path, 'images')
        
        frames = os.listdir(rgb_folder_path)
        # Sort the frames by name
        frames.sort()

        
        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")
        

        start_idx = frame_idx



        selected_frames = frames[start_idx:start_idx + self.sample_frames]


        while check_all_files_exist(selected_frames, rgb_folder_path) == False:
            start_idx = random.randint(0, len(frames) - self.sample_frames)
            selected_frames = frames[start_idx:start_idx + self.sample_frames]







        return {'rgb_folder_path': rgb_folder_path, 'start_idx': start_idx}