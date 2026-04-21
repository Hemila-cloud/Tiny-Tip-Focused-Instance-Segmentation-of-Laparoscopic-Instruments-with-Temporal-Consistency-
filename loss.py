import tensorflow as tf

def weighted_binary_crossentropy(weight=3):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Compute standard BCE per pixel (no reduction)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # Create pixel-wise weights
        weights = 1 + (weight - 1) * y_true

        # Apply weights
        weighted_bce = weights * bce

        return tf.reduce_mean(weighted_bce)

    return loss


def dice_coef(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def combined_loss(y_true, y_pred):
    smooth = 1e-6
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # BCE
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    # Dice
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    
    dice_loss = 1 - dice

    return bce + 2 * dice_loss   # give more weight to Dice

