import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src.backend.config import floatx
from keras.src.backend.openvino import numpy as ov_numpy
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import convert_to_numpy
from keras.src.backend.openvino.core import get_ov_output
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed.data)
    normal_const = rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    return OpenVINOKerasTensor(ov_opset.constant(normal_const).output(0))


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed_val = draw_seed(seed)
    if isinstance(seed_val, OpenVINOKerasTensor):
        seed_data = convert_to_numpy(seed_val)
    else:
        seed_data = seed_val.data
    rng = np.random.default_rng(seed_data)
    random_values = rng.uniform(minval, maxval, size=shape).astype(dtype)
    return OpenVINOKerasTensor(ov_opset.constant(random_values).output(0))


def categorical(logits, num_samples, dtype="int64", seed=None):
    dtype = dtype or "int64"
    ov_dtype = OPENVINO_DTYPES[dtype]
    logits = get_ov_output(logits)

    zero_const = ov_opset.constant(0, Type.i32).output(0)
    one_const = ov_opset.constant(1, Type.i32).output(0)
    neg_one_const = ov_opset.constant(-1, Type.i32).output(0)

    # Compute probabilities and cumulative sum
    probs = ov_opset.softmax(logits, axis=-1).output(0)
    cumsum_probs = ov_opset.cumsum(probs, neg_one_const).output(0)

    # Get shape and compute batch dimensions
    logits_shape = ov_opset.shape_of(logits, Type.i32).output(0)
    rank = ov_opset.shape_of(logits_shape, Type.i32).output(0)
    rank_scalar = ov_opset.squeeze(rank, zero_const).output(0)
    rank_minus_1 = ov_opset.subtract(rank_scalar, one_const).output(0)

    # Extract batch shape (all dimensions except last)
    batch_indices = ov_opset.range(
        zero_const, rank_minus_1, one_const, output_type=Type.i32
    ).output(0)
    batch_shape = ov_opset.gather(logits_shape, batch_indices, axis=0).output(0)

    # Create final shape [batch_dims..., num_samples]
    num_samples_const = ov_opset.constant([num_samples], Type.i32).output(0)
    final_shape = ov_opset.concat(
        [batch_shape, num_samples_const], axis=0
    ).output(0)

    seed_tensor = draw_seed(seed)
    if isinstance(seed_tensor, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_tensor)
    else:
        seed1, seed2 = seed_tensor.data

    probs_dtype = probs.get_element_type()
    zero_float = ov_opset.constant(0.0, probs_dtype).output(0)
    one_float = ov_opset.constant(1.0, probs_dtype).output(0)

    rand = ov_opset.random_uniform(
        final_shape, zero_float, one_float, probs_dtype, seed1, seed2
    ).output(0)

    rand_unsqueezed = ov_opset.unsqueeze(rand, neg_one_const).output(0)
    cumsum_unsqueezed = ov_opset.unsqueeze(cumsum_probs, one_const).output(0)

    # Count how many cumulative probabilities each random number exceeds
    greater = ov_opset.greater(rand_unsqueezed, cumsum_unsqueezed).output(0)
    samples = ov_opset.reduce_sum(
        ov_opset.convert(greater, Type.i32).output(0), neg_one_const
    ).output(0)

    result = ov_opset.convert(samples, ov_dtype).output(0)
    return OpenVINOKerasTensor(result)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    dtype = dtype or "int32"
    ov_dtype = OPENVINO_DTYPES[dtype]
    seed_val = draw_seed(seed)
    if isinstance(seed_val, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_val)
    else:
        seed1, seed2 = seed_val.data
    if ov_dtype in (Type.i64, Type.u64, Type.u32):
        gen_dtype = Type.i64
    else:
        gen_dtype = Type.i32
    if isinstance(shape, (list, tuple)):
        shape = ov_opset.constant(list(shape), Type.i32).output(0)
    elif isinstance(shape, OpenVINOKerasTensor):
        shape = shape.output
    elif isinstance(shape, int):
        shape = ov_opset.constant([shape], Type.i32).output(0)
    else:
        shape = get_ov_output(shape, Type.i32)
    minval = get_ov_output(minval, gen_dtype)
    maxval = get_ov_output(maxval, gen_dtype)
    if minval.get_element_type() != gen_dtype:
        minval = ov_opset.convert(minval, gen_dtype).output(0)
    if maxval.get_element_type() != gen_dtype:
        maxval = ov_opset.convert(maxval, gen_dtype).output(0)
    rand = ov_opset.random_uniform(
        shape, minval, maxval, gen_dtype, seed1, seed2
    ).output(0)
    if ov_dtype != gen_dtype:
        result = ov_opset.convert(rand, ov_dtype).output(0)
    else:
        result = rand
    return OpenVINOKerasTensor(result)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed.data)

    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev

    flat_shape = np.prod(shape)
    random_numbers = np.empty(0)

    # loop until we have enough valid numbers to fill our desired shape
    while random_numbers.shape[0] < flat_shape:
        # Generate a batch of random numbers from a normal distribution
        batch = rng.normal(loc=mean, scale=stddev, size=flat_shape)

        # Filter the numbers to keep only those within the specified bounds
        valid = batch[(batch >= lower_bound) & (batch <= upper_bound)]

        # Append the valid numbers to the result array
        random_numbers = np.append(random_numbers, valid)

    # Truncate the result array to the desired size and reshape it
    np_array_res = random_numbers[:flat_shape].astype(dtype).reshape(shape)
    return OpenVINOKerasTensor(ov_opset.constant(np_array_res).output(0))


def dropout(inputs, rate, noise_shape=None, seed=None):
    raise NotImplementedError(
        "`dropout` is not supported with openvino backend"
    )


def shuffle(x, axis=0, seed=None):
    seed_tensor = draw_seed(seed)
    if isinstance(seed_tensor, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_tensor)
    else:
        seed1, seed2 = seed_tensor.data
    x_ov = get_ov_output(x)
    x_shape = x_ov.get_partial_shape()
    rank = x_shape.rank.get_length()
    if axis < 0:
        axis += rank
    shape_tensor = ov_opset.shape_of(x_ov, Type.i32).output(0)
    dim_size = ov_opset.gather(
        shape_tensor,
        ov_opset.constant([axis], Type.i32).output(0),
        ov_opset.constant(0, Type.i32).output(0),
    ).output(0)
    min_val = ov_opset.constant(0.0, Type.f32).output(0)
    max_val = ov_opset.constant(1.0, Type.f32).output(0)
    rand_shape = ov_opset.reshape(
        dim_size, ov_opset.constant([1], Type.i32).output(0), False
    ).output(0)
    rand_values = ov_opset.random_uniform(
        rand_shape, min_val, max_val, Type.f32, seed1, seed2
    ).output(0)
    indices = ov_numpy.argsort(OpenVINOKerasTensor(rand_values), axis=0)
    return ov_numpy.take(x, indices, axis=axis)


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError("`gamma` is not supported with openvino backend")


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    ov_dtype = OPENVINO_DTYPES[dtype]
    seed_val = draw_seed(seed)
    if isinstance(seed_val, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_val)
    else:
        seed1, seed2 = seed_val.data
    counts = get_ov_output(counts)
    probabilities = get_ov_output(probabilities)
    calc_dtype = Type.f32
    counts_f = ov_opset.convert(counts, calc_dtype).output(0)
    probs_f = ov_opset.convert(probabilities, calc_dtype).output(0)
    if isinstance(shape, (list, tuple)):
        shape_tensor = ov_opset.constant(list(shape), Type.i32).output(0)
    elif isinstance(shape, OpenVINOKerasTensor):
        shape_tensor = shape.output
    else:
        shape_tensor = get_ov_output(shape, Type.i32)
    zero = ov_opset.constant(0.0, calc_dtype).output(0)
    one = ov_opset.constant(1.0, calc_dtype).output(0)
    u1 = ov_opset.random_uniform(
        shape_tensor, zero, one, calc_dtype, seed1, seed2
    ).output(0)
    u2 = ov_opset.random_uniform(
        shape_tensor, zero, one, calc_dtype, seed1, seed2 + 1
    ).output(0)
    epsilon = 1e-7
    epsilon_const = ov_opset.constant(epsilon, calc_dtype).output(0)
    u1_safe = ov_opset.maximum(u1, epsilon_const).output(0)
    log_u1 = ov_opset.log(u1_safe).output(0)
    neg_two = ov_opset.constant(-2.0, calc_dtype).output(0)
    two_pi = ov_opset.constant(2 * np.pi, calc_dtype).output(0)
    r = ov_opset.sqrt(ov_opset.multiply(neg_two, log_u1)).output(0)
    theta = ov_opset.multiply(two_pi, u2).output(0)
    z = ov_opset.multiply(r, ov_opset.cos(theta)).output(0)
    mean = ov_opset.multiply(counts_f, probs_f).output(0)
    one_minus_p = ov_opset.subtract(one, probs_f).output(0)
    var = ov_opset.multiply(mean, one_minus_p).output(0)
    std = ov_opset.sqrt(var).output(0)
    res_normal = ov_opset.add(mean, ov_opset.multiply(std, z)).output(0)
    res_normal = ov_opset.round(res_normal, mode="half_to_even").output(0)
    res_normal = ov_opset.maximum(res_normal, zero).output(0)
    res_normal = ov_opset.minimum(res_normal, counts_f).output(0)
    is_one = ov_opset.equal(counts_f, one).output(0)
    bernoulli = ov_opset.less(u1, probs_f).output(0)
    bernoulli_f = ov_opset.convert(bernoulli, calc_dtype).output(0)
    res = ov_opset.select(is_one, bernoulli_f, res_normal).output(0)
    if ov_dtype != calc_dtype:
        res = ov_opset.convert(res, ov_dtype).output(0)
    return OpenVINOKerasTensor(res)



def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError("`beta` is not supported with openvino backend")
