class MockOp:
    def __init__(self, shape):
        self.shape = shape
    def output(self, idx):
        return self
    def get_partial_shape(self):
        from collections import namedtuple
        Shape = namedtuple('Shape', ['rank'])
        return Shape(len(self.shape))

class MockOpset:
    def gather(self, data, indices, axis):
        if indices == 0 or indices == 1:
            return MockOp(data.shape[1:])
        return MockOp(data.shape)

    def divide(self, a, b):
        return MockOp(a.shape)

    def multiply(self, a, b):
        return MockOp(a.shape)

    def shape_of(self, data, t):
        return MockOp([len(data.shape)])

    def constant(self, val, dtype=None):
        if isinstance(val, list):
            return MockOp([len(val)])
        return val

    def slice(self, data, start, stop, step):
        # mock slice
        return MockOp([stop - start] if isinstance(start, int) else [0]) # fake

    def add(self, a, b):
        return MockOp(a.shape)

    def concat(self, lst, axis):
        return MockOp([sum(x.shape[0] for x in lst)])

    def unsqueeze(self, data, axis):
        new_shape = list(data.shape)
        new_shape.insert(0, 1)
        return MockOp(new_shape)

ov_opset = MockOpset()

bias = MockOp([2, 9])

bias_W = ov_opset.gather(
    bias,
    0,
    0,
)
print("bias_W shape:", bias_W.shape)
