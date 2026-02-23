
import openvino as ov
import openvino.opset15 as ov_opset
from openvino import Type
import numpy as np

def test_random_uniform_symbolic_seed():
    shape = ov_opset.constant([5], Type.i32)
    min_val = ov_opset.constant(0.0, Type.f32)
    max_val = ov_opset.constant(1.0, Type.f32)

    # Create parameter for seed
    seed1 = ov_opset.parameter(shape=[], dtype=Type.u64)
    seed2 = ov_opset.parameter(shape=[], dtype=Type.u64)

    try:
        rand = ov_opset.random_uniform(shape, min_val, max_val, Type.f32, seed1.output(0), seed2.output(0))
        print("Success: random_uniform accepts symbolic seeds")
    except Exception as e:
        print(f"Failure: {e}")

if __name__ == "__main__":
    test_random_uniform_symbolic_seed()
