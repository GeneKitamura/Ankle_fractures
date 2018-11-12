import tensorflow as tf

def add_collection(name, op):
    tf.add_to_collection(name, op)

def activation_summary(inp_tensor):
    tf.summary.histogram('activation', inp_tensor)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(inp_tensor))