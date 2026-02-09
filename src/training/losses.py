import tensorflow as tf
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)

    return (2. * intersection + smooth) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tumor_dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_tumor = y_true[..., 1:] 
    y_pred_tumor = y_pred[..., 1:]
    
    axes = (0, 1, 2, 3)
    intersection = K.sum(y_true_tumor * y_pred_tumor, axis=axes)
    union = K.sum(y_true_tumor, axis=axes) + K.sum(y_pred_tumor, axis=axes)
    
    return K.mean((2. * intersection + smooth) / (union + smooth))

def categorical_dice_loss(y_true, y_pred, smooth=1e-6):
    # Without class 0
    y_true_tumor = y_true[..., 1:]
    y_pred_tumor = y_pred[..., 1:]
    
    axes = (0, 1, 2, 3)
    
    intersection = K.sum(y_true_tumor * y_pred_tumor, axis=axes)
    union = K.sum(y_true_tumor, axis=axes) + K.sum(y_pred_tumor, axis=axes)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - K.mean(dice)