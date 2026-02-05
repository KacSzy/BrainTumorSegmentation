import tensorflow as tf
import tensorflow.keras.backend as bck

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_flatten = bck.flatten(y_true)
    y_pred_flatten = bck.flatten(y_pred)
    intersection = bck.sum(y_true_flatten * y_pred_flatten)

    return (2. * intersection + smooth) / (bck.sum(y_true_flatten) + bck.sum(y_pred_flatten) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
