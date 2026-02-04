import os
import numpy as np
import nibabel as nib
from typing import Tuple, List


class BraTSLoader:
    """
    A simple loader for Brain Tumor Segmentation (BraTS) dataset.

    This class handles loading of multi-modal MRI scans and segmentation
    masks for brain tumor analysis.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError("Provided data directory does not exist.")

    def get_patient_ids(self) -> List[str]:
        """Returns a list of patient IDs available in the dataset."""
        return [d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))]

    def load_patient_data(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the MRI scans and segmentation mask for a given `patient`.

        :param patient_id: The ID of the patient to load data for. Must match a directory name in the dataset.
        :return: A tuple containing the 4-channel MRI volume (H, W, D, 4) and the segmentation mask (H, W, D).
        
        :raises FileNotFoundError: If a required modality file is missing.
        """
        patient_path = os.path.join(self.data_dir, patient_id)

        if not os.path.isdir(patient_path):
            raise FileNotFoundError("Patient folder not found.")

        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []

        # 1. Loading MRI Modalities
        for mod in modalities:

            file_name = f"{patient_id}_{mod}.nii"
            file_path = os.path.join(patient_path, file_name)

            img = nib.load(file_path).get_fdata().astype(np.float32)
            slices.append(img)

        # Stack to (H, W, D, C)
        volume = np.stack(slices, axis=-1)

        # 2. Loading Segmentation Mask
        mask_name = f"{patient_id}_seg.nii"
        mask_path = os.path.join(patient_path, mask_name)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        return volume, mask
