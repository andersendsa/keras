import openvino.opset15 as ov_opset
from openvino import Type
import numpy as np

bias = ov_opset.constant(np.ones((2, 9), dtype=np.float32)).output(0)

# Keras bias shape: [2, 3 * hidden_size]
bias_W = ov_opset.gather(
    bias,
    ov_opset.constant([0], dtype=Type.i32).output(0),
    ov_opset.constant(0, dtype=Type.i32).output(0),
).output(0)
bias_R = ov_opset.gather(
    bias,
    ov_opset.constant([1], dtype=Type.i32).output(0),
    ov_opset.constant(0, dtype=Type.i32).output(0),
).output(0)
shape_dim = ov_opset.shape_of(bias_W, Type.i32).output(0)
units = ov_opset.divide(
    ov_opset.gather(
        shape_dim,
        ov_opset.constant([0], dtype=Type.i32).output(0),
        ov_opset.constant(0, dtype=Type.i32).output(0),
    ).output(0),
    ov_opset.constant(3, dtype=Type.i32).output(0),
).output(0)
units_x2 = ov_opset.multiply(
    units, ov_opset.constant(2, dtype=Type.i32).output(0)
).output(0)

Wb_zr = ov_opset.slice(
    bias_W,
    ov_opset.constant([0], dtype=Type.i32).output(0),
    units_x2,
    ov_opset.constant([1], dtype=Type.i32).output(0),
).output(0)
Wb_h = ov_opset.slice(
    bias_W,
    units_x2,
    ov_opset.constant([2147483647], dtype=Type.i32).output(0),
    ov_opset.constant([1], dtype=Type.i32).output(0),
).output(0)

Rb_zr = ov_opset.slice(
    bias_R,
    ov_opset.constant([0], dtype=Type.i32).output(0),
    units_x2,
    ov_opset.constant([1], dtype=Type.i32).output(0),
).output(0)
Rb_h = ov_opset.slice(
    bias_R,
    units_x2,
    ov_opset.constant([2147483647], dtype=Type.i32).output(0),
    ov_opset.constant([1], dtype=Type.i32).output(0),
).output(0)

Wb_zr_plus_Rb_zr = ov_opset.add(Wb_zr, Rb_zr).output(0)

bias_ov = ov_opset.concat(
    [Wb_zr_plus_Rb_zr, Wb_h, Rb_h], axis=0
).output(0)

print("bias_ov shape before unsqueeze:", bias_ov.get_partial_shape())
bias_ov = ov_opset.unsqueeze(
    bias_ov, ov_opset.constant([0], dtype=Type.i32).output(0)
).output(0)
print("bias_ov shape after unsqueeze:", bias_ov.get_partial_shape())
