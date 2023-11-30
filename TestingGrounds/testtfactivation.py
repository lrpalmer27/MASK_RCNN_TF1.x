import tensorflow as tf

print("\n\n ------------------------------\n\n")
tf.device("/gpu:0")
tf.test.is_gpu_available()