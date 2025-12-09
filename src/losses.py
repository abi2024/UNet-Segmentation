import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    """
    A metric that calculates the overlap.
    1.0 = Perfect overlap
    0.0 = No overlap
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Smooth adds stability (prevents division by zero)
    smooth = 1.0 
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Loss function is what we minimize.
    Since we want to MAXIMIZE Dice Score, we minimize (1 - Dice Score).
    """
    return 1.0 - dice_coef(y_true, y_pred)