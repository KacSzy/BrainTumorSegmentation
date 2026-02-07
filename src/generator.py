# src/data_generator.py
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class BraTSGenerator(tf.keras.utils.Sequence):
    """
    A data generator for Brain Tumor Segmentation (BraTS) dataset.

    This class implements a Keras Sequence generator that loads preprocessed
    MRI volumes and segmentation masks in batches during model training.
    """

    def __init__(self, patient_ids, data_dir, batch_size, img_size=(128, 128, 128), shuffle=True, **kwargs):
        """
        Initializes the BraTS data generator.

        :param patient_ids: List of patient IDs to generate data for.
        :param data_dir: Directory containing preprocessed patient data.
        :param batch_size: Number of samples per batch.
        :param img_size: Expected dimensions of input volumes (H, W, D). Defaults to (128, 128, 128).
        :param shuffle: Whether to shuffle patient IDs at the end of each epoch. Defaults to True.
        """
        super().__init__(**kwargs)

        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.patient_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data.

        :param index: Batch index.
        :return: Tuple of (X, y) where X is the input volume batch and y is the segmentation mask batch.
        """
        # Generate indexes for the current batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_ids = [self.patient_ids[k] for k in indexes]

        # Generate data for the batch
        X, y = self.__data_generation(batch_ids)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch and shuffles if enabled."""
        self.indexes = np.arange(len(self.patient_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        """
        Generates data for a batch of patient IDs.

        :param batch_ids: List of patient IDs for the current batch.
        :return: Tuple containing the batch of MRI volumes (B, H, W, D, 4) and one-hot encoded masks (B, H, W, D, n_classes).

        :raises FileNotFoundError: If volume or mask files are missing for a patient.
        """
        # Initialize empty arrays for batch
        X = np.empty((self.batch_size, *self.img_size, 4), dtype=np.float32)
        y = np.empty((self.batch_size, *self.img_size, 4), dtype=np.float32)

        # Load data for each patient in the batch
        for i, ID in enumerate(batch_ids):

            patient_dir = os.path.join(self.data_dir, ID)

            vol_path = os.path.join(patient_dir, 'volume.nii')
            mask_path = os.path.join(patient_dir, 'mask.nii')

            # Load preprocessed volume and mask
            vol = nib.load(vol_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()

            X[i,] = vol

            # Convert mask to one-hot encoding
            y[i,] = to_categorical(mask, num_classes=4)

        return X, y
