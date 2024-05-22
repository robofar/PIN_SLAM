#import math
import os
import glob

#import cv2
import numpy as np
import torch
#import json
#import imageio

#class BaseDataset(torch.utils.data.Dataset):


# Since I have only KITTI dataset, I will not create BaseDataset since I dont know in which
# format are for example calibration files for other datasets
# I will just leave KITTIDataset for now only

class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict):
        self.config = config

        self.dataset_path = config["Dataset"]["dataset_path"]
        self.poses_path = config["Dataset"]["poses_path"]
        self.calibration_path = config["Dataset"]["calibration_path"]

        self.load_kitti_paths(self.dataset_path)
        self.load_kitti_poses(self.poses_path)
        self.load_kitti_calibration(self.calibration_path)

        
    def load_kitti_paths(self, dataset_path):
        self.color_paths = sorted(glob.glob(os.path.join(dataset_path, '*.png')))
        self.num_frames = len(self.color_paths)

    def load_kitti_poses(self, poses_path):
        """Load ground truth poses (T_w_cam0) from file."""
        poses = []
        try:
            with open(poses_path, 'r') as f:
                lines = f.readlines()

                # TODO: add option to select only subset of frames (if you dont want whole sequence)

                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not available for sequence ' + self.sequence + '.')

        self.poses = poses

    def load_kitti_calibration(self, calibration_path):
        pass

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # here actual image,dept and pose will be returned

        return 1,2,3



def load_dataset(config):
    if config["Dataset"]["type"] == "KITTI":
        return KITTIDataset(config)
    else:
        raise ValueError("Unknown dataset type [for now]")