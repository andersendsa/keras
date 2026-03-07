class MockType:
    i32 = "i32"

class MockOutput:
    def __init__(self, shape):
        self.shape = shape
    def output(self, idx):
        return self
    def get_partial_shape(self):
        class Shape:
            def __init__(self, shp):
                self.shp = shp
                self.rank = type('Rank', (), {'get_length': lambda self=None: len(shp)})()
            def __str__(self):
                return str(self.shp)
        return Shape(self.shape)

class MockOpset:
    def constant(self, val, dtype=None):
        if isinstance(val, int):
            return MockOutput([])
        elif isinstance(val, list):
            return MockOutput([len(val)])
        elif hasattr(val, 'shape'):
            return MockOutput(list(val.shape))
        return MockOutput([])

    def gather(self, data, indices, axis):
        shape = list(data.shape)
        ax = axis.shape[0] if len(axis.shape) > 0 else 0
        idx_len = indices.shape[0] if len(indices.shape) > 0 else None

        if idx_len is not None:
            # 1D index array: keeps the dimension, changes size
            shape[ax] = idx_len
        else:
            # scalar index: removes the dimension
            shape.pop(ax)
        return MockOutput(shape)

    def shape_of(self, data, dtype):
        return MockOutput([len(data.shape)])

    def divide(self, a, b):
        return MockOutput(a.shape)

    def multiply(self, a, b):
        return MockOutput(a.shape)

    def slice(self, data, start, stop, step):
        shape = list(data.shape)
        return MockOutput(shape)

    def add(self, a, b):
        return MockOutput(a.shape)

    def concat(self, inputs, axis):
        shape = list(inputs[0].shape)
        # simplistic concat sim for our mock
        return MockOutput(shape)

    def unsqueeze(self, data, axes):
        shape = list(data.shape)
        shape.insert(0, 1) # hardcode axis 0 for mock
        return MockOutput(shape)

ov_opset = MockOpset()
Type = MockType()
class NumpySim:
    def ones(self, shape, dtype):
        class A:
            def __init__(self, s): self.shape = s
        return A(shape)
np = NumpySim()
np.float32 = "f32"

# Simulate reset_after=True bias input [2, 9] (units=3)
bias = ov_opset.constant(np.ones((2, 9), dtype=np.float32)).output(0)

# SCALAR INDICES FIX:
bias_W = ov_opset.gather(
    bias,
    ov_opset.constant(0, dtype=Type.i32).output(0),
    ov_opset.constant(0, dtype=Type.i32).output(0),
).output(0)

bias_R = ov_opset.gather(
    bias,
    ov_opset.constant(1, dtype=Type.i32).output(0),
    ov_opset.constant(0, dtype=Type.i32).output(0),
).output(0)

shape_dim = ov_opset.shape_of(bias_W, Type.i32).output(0)
units = ov_opset.divide(
    ov_opset.gather(
        shape_dim,
        ov_opset.constant(0, dtype=Type.i32).output(0),
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

print(f"bias_ov rank BEFORE unsqueeze: {bias_ov.get_partial_shape().rank.get_length()}")

bias_ov = ov_opset.unsqueeze(
    bias_ov, ov_opset.constant([0], dtype=Type.i32).output(0)
).output(0)

print(f"bias_ov rank AFTER unsqueeze: {bias_ov.get_partial_shape().rank.get_length()}")
print(f"bias_ov shape: {bias_ov.get_partial_shape()}")
