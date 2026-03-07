import openvino.opset15 as ov_opset
from openvino import Type
import numpy as np

bias = ov_opset.constant(np.ones((2, 9), dtype=np.float32)).output(0)
# This simulates Keras bias for reset_after=True, units=3

bias_W = ov_opset.gather(
    bias,
    ov_opset.constant([0], dtype=Type.i32).output(0),
    ov_opset.constant(0, dtype=Type.i32).output(0),
).output(0)
print("bias_W rank:", bias_W.get_partial_shape().rank.get_length())
print("bias_W shape:", bias_W.get_partial_shape())
