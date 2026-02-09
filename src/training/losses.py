import tensorflow as tf
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice coefficient between ground truth and predictions.

    The Dice coefficient measures the overlap between two samples and ranges from 0 to 1,
    where 1 indicates perfect overlap.

    :param y_true: Ground truth tensor.
    :param y_pred: Predicted tensor.
    :param smooth: Smoothing factor to avoid division by zero (default: 1e-6).
    :return: Dice coefficient value.
    """
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)

    return (2. * intersection + smooth) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) + smooth)

def dice_loss(y_true, y_pred):
    """
    Calculates the Dice loss function.

    Dice loss is defined as 1 minus the Dice coefficient, making it suitable
    for minimization during training.

    :param y_true: Ground truth tensor.
    :param y_pred: Predicted tensor.
    :return: Dice loss value.
    """
    return 1 - dice_coef(y_true, y_pred)

def tumor_dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice coefficient for tumor classes only.

    This function excludes the background class (channel 0) and computes
    the average Dice coefficient across all tumor classes.

    :param y_true: Ground truth tensor with shape (batch, height, width, depth, classes).
    :param y_pred: Predicted tensor with shape (batch, height, width, depth, classes).
    :param smooth: Smoothing factor to avoid division by zero (default: 1e-6).
    :return: Mean Dice coefficient across tumor classes.
    """
    y_true_tumor = y_true[..., 1:] 
    y_pred_tumor = y_pred[..., 1:]
    
    axes = (0, 1, 2, 3)
    intersection = K.sum(y_true_tumor * y_pred_tumor, axis=axes)
    union = K.sum(y_true_tumor, axis=axes) + K.sum(y_pred_tumor, axis=axes)
    
    return K.mean((2. * intersection + smooth) / (union + smooth))

def categorical_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Calculates the categorical Dice loss for multi-class segmentation.

    This loss excludes the background class (channel 0) and computes the
    average Dice loss across all tumor classes.

    :param y_true: Ground truth tensor with shape (batch, height, width, depth, classes).
    :param y_pred: Predicted tensor with shape (batch, height, width, depth, classes).
    :param smooth: Smoothing factor to avoid division by zero (default: 1e-6).
    :return: Categorical Dice loss value.
    """
    y_true_tumor = y_true[..., 1:]
    y_pred_tumor = y_pred[..., 1:]
    
    axes = (0, 1, 2, 3)
    
    intersection = K.sum(y_true_tumor * y_pred_tumor, axis=axes)
    union = K.sum(y_true_tumor, axis=axes) + K.sum(y_pred_tumor, axis=axes)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - K.mean(dice)

def combined_loss(y_true, y_pred):
    """
    Calculates a combined loss function using Dice loss and categorical cross-entropy.

    This loss combines categorical Dice loss with categorical cross-entropy to leverage
    both region-based and pixel-wise optimization objectives.

    :param y_true: Ground truth tensor with shape (batch, height, width, depth, classes).
    :param y_pred: Predicted tensor with shape (batch, height, width, depth, classes).
    :return: Combined loss value (sum of Dice loss and categorical cross-entropy).
    """
    loss_dice = categorical_dice_loss(y_true, y_pred)

    loss_ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss_ce = K.mean(loss_ce)

    return loss_dice + loss_ce
