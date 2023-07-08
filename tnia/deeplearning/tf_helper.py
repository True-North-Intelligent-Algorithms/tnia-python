import tensorflow as tf

def force_cpu():
    tf.config.set_visible_devices([], 'GPU')
