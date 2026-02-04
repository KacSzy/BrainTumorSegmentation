
# KROK 4: Ustalenie stałego rozmiaru (Resize / Padding)
# Po Kroku 2 (Crop) każdy pacjent będzie miał inny rozmiar "pudełka" z mózgiem. Sieci neuronowe (zazwyczaj) wymagają stałego wejścia, żeby łączyć dane w Batche.
# Decyzja: Musisz sprowadzić wszystko do jednego wymiaru, np. (128, 128, 128).
# Jak to zrobić:Opcja A (Resize): Interpolacja obrazu (może zniekształcić guza).
# Opcja B (Crop/Pad - Zalecane): Jeśli mózg jest mniejszy niż 128 – doklej zera (padding). Jeśli większy – przytnij (ale uważaj, żeby nie wyciąć guza!).
# Pro tip na start: Na początek możesz po prostu przyciąć wszystko "na sztywno" do środka, np. biorąc środkowe 128x128x128 wokseli.
# To najprostsze, choć ryzykowne (guz może być na brzegu).
import numpy as np


def _fix_label_mapping(mask) -> np.ndarray:
    return np.where(mask == 4, 3, mask)


def _remove_background(volume, mask) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = np.where(volume > 0)[0].min(), np.where(volume > 0)[0].max()
    y_min, y_max = np.where(volume > 0)[1].min(), np.where(volume > 0)[1].max()
    z_min, z_max = np.where(volume > 0)[2].min(), np.where(volume > 0)[2].max()

    return volume[x_min:x_max, y_min:y_max, z_min:z_max], mask[x_min:x_max, y_min:y_max, z_min:z_max]


def _normalize(volume: np.ndarray) -> np.ndarray:
    mask = np.any(volume > 0, axis=-1)

    epsilon = 1e-7

    for channel in range(volume.shape[3]):
        brain_voxels = volume[mask, channel]

        mean = brain_voxels.mean()
        std = brain_voxels.std()

        volume[:, :, :, channel] = (volume[:, :, :, channel] - mean) / (std + epsilon)

    return volume


def _crop_to_target_shape(volume, mask, target_shape) -> tuple[np.ndarray, np.ndarray]:
    # 1. Padding
    padding = [
                    (0, max(0, target_shape[i] - volume.shape[i])) for i in range(3)
                ] + [(0, 0)]

    volume = np.pad(volume, padding)
    mask = np.pad(mask, padding[:-1])

    if volume.shape[:3] == target_shape:
        return volume, mask

    # 2. Cropping
    tumor_coords = np.argwhere(mask > 0)

    current_shape = volume.shape[:3] # (H, W, D)

    # center on tumor if present
    if len(tumor_coords) > 0:
        center = tumor_coords.mean(axis=0).astype(int)

    else:
        center = np.array(current_shape) // 2

    starts = [curr_center - target // 2 for curr_center, target in zip(center, target_shape)]

    # 3. Clamping

    final_slices = []
    for i in range(3):
        start = starts[i]
        dim_len = current_shape[i]
        target = target_shape[i]

        # If start is negative, set it to 0
        start = max(0, start)

        # if start + target is larger than dim_len, shift start back so that the end aligns with dim_len
        if start + target > dim_len:
            start = dim_len - target

        final_slices.append(slice(start, start + target))

    x_slice, y_slice, z_slice = final_slices

    return volume[x_slice, y_slice, z_slice, :], mask[x_slice, y_slice, z_slice]


def preprocess_data(volume, mask, target_shape=(128, 128, 128)) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses MRI by applying a series of transformations to the input volume and mask.

    Steps:
    1. Fix label mapping in the mask (4 -> 3).
    2. Remove the background from the `volume` and `mask` to focus on the brain region.
    3. Normalize the volume using Z-Score normalization.
    4. Crop the volume and mask to a `target shape`.

    :param volume: The input 3D MRI volume data.
    :param mask: The corresponding mask for the volume, containing labeled regions.
    :param target_shape: The desired output shape for the volume and mask.
    :return: Prepared data for analyzing brain tumors.
    """

    mask = _fix_label_mapping(mask)
    volume, mask = _remove_background(volume, mask)
    volume = _normalize(volume)
    processed_volume, processed_mask = _crop_to_target_shape(volume, mask, target_shape)

    return processed_volume, processed_mask
