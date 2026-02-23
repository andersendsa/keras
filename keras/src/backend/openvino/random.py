import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src.backend.config import floatx
from keras.src.backend.openvino import numpy as ov_numpy
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import convert_to_numpy
from keras.src.backend.openvino.core import get_ov_output
from keras.src.backend.openvino.core import while_loop
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def _int64_arithmetic_rshift(x, shift_val):
    # Implements arithmetic right shift using integer division to avoid BitwiseRightShift issues
    # x >> n is floor(x / 2^n)

    # shift_val is integer (python).
    divisor_val = 1 << shift_val
    divisor = ov_opset.constant(np.array([divisor_val], dtype=np.int64)).output(0)

    q = ov_opset.divide(x, divisor).output(0) # Truncated division
    r = ov_opset.mod(x, divisor).output(0)    # Truncated modulo

    # If x < 0 and r != 0, subtract 1 from q
    zero = ov_opset.constant(np.array([0], dtype=np.int64)).output(0)
    one = ov_opset.constant(np.array([1], dtype=np.int64)).output(0)

    is_neg = ov_opset.less(x, zero).output(0)
    has_rem = ov_opset.not_equal(r, zero).output(0)
    cond = ov_opset.logical_and(is_neg, has_rem).output(0)

    result = ov_opset.select(cond, ov_opset.subtract(q, one), q).output(0)
    return result


def _int64_logical_and_mask(x, mask_bits):
    # Implements x & mask where mask = (1 << mask_bits) - 1
    # This is equivalent to positive_mod(x, 1 << mask_bits)
    divisor_val = 1 << mask_bits
    divisor = ov_opset.constant(np.array([divisor_val], dtype=np.int64)).output(0)

    # r = x % divisor (truncated)
    r = ov_opset.mod(x, divisor).output(0)

    # positive_mod = (r + divisor) % divisor
    res = ov_opset.mod(ov_opset.add(r, divisor), divisor).output(0)
    return res


def _hash_int64(k):
    # SplitMix64

    # 0xbf58476d1ce4e5b9 = -4658895280553007687
    m1 = ov_opset.constant(np.array([-4658895280553007687], dtype=np.int64)).output(0)
    # 0x94d049bb133111eb = -7723592293110705685
    m2 = ov_opset.constant(np.array([-7723592293110705685], dtype=np.int64)).output(0)

    # k = (k + ((k >> 30) & mask)) * m1
    shifted = _int64_arithmetic_rshift(k, 30)
    shifted = _int64_logical_and_mask(shifted, 64 - 30)
    k = ov_opset.add(k, shifted).output(0)
    k = ov_opset.multiply(k, m1).output(0)

    # Squaring for non-linearity
    k = ov_opset.multiply(k, k).output(0)

    # k = (k + ((k >> 27) & mask)) * m2
    shifted = _int64_arithmetic_rshift(k, 27)
    shifted = _int64_logical_and_mask(shifted, 64 - 27)
    k = ov_opset.add(k, shifted).output(0)
    k = ov_opset.multiply(k, m2).output(0)

    # k = k + ((k >> 31) & mask)
    shifted = _int64_arithmetic_rshift(k, 31)
    shifted = _int64_logical_and_mask(shifted, 64 - 31)
    k = ov_opset.add(k, shifted).output(0)
    return k


def _stateless_uniform(shape, seed, dtype=None):
    # Implements a counter-based PRNG using SplitMix64 on int64
    dtype = dtype or floatx()
    target_type = OPENVINO_DTYPES[dtype]

    seed = get_ov_output(seed)
    # Convert seed to int64
    seed_i64 = ov_opset.convert(seed, Type.i64).output(0)

    key = ov_opset.gather(seed_i64, ov_opset.constant(0, Type.i32), ov_opset.constant(0, Type.i32)).output(0)
    key = ov_opset.reshape(key, ov_opset.constant([1], Type.i32).output(0), False).output(0)

    counter = ov_opset.gather(seed_i64, ov_opset.constant(1, Type.i32), ov_opset.constant(0, Type.i32)).output(0)
    counter = ov_opset.reshape(counter, ov_opset.constant([1], Type.i32).output(0), False).output(0)

    shape_tensor = get_ov_output(shape, Type.i32)
    if not isinstance(shape, (list, tuple)):
         shape_tensor = ov_opset.convert(shape_tensor, Type.i32).output(0)

    num_elements = ov_opset.reduce_prod(shape_tensor, ov_opset.constant(0, Type.i32), keep_dims=False).output(0)
    # num_elements is i32, convert to i64 for range
    num_elements_i64 = ov_opset.convert(num_elements, Type.i64).output(0)

    indices = ov_opset.range(
        ov_opset.constant(0, Type.i64).output(0),
        num_elements_i64,
        ov_opset.constant(1, Type.i64).output(0),
        Type.i64
    ).output(0)

    inputs = ov_opset.add(indices, counter).output(0)
    inputs = ov_opset.add(inputs, key).output(0)

    hashed = _hash_int64(inputs)

    # Mask to lower 24 bits for float conversion (f32)
    # Using modulo instead of bitwise_and to avoid constant folding issues
    hashed_masked = _int64_logical_and_mask(hashed, 24)
    hashed_float = ov_opset.convert(hashed_masked, target_type).output(0)

    # Divide by 2^24 = 16777216.0
    scale = ov_opset.constant(1.0 / 16777216.0, target_type).output(0)
    random_values = ov_opset.multiply(hashed_float, scale).output(0)

    random_values = ov_opset.reshape(random_values, shape_tensor, False).output(0)

    # Update seed counter
    increment = ov_opset.add(num_elements_i64, ov_opset.constant(12345, Type.i64).output(0)).output(0)

    new_counter = ov_opset.add(
        ov_opset.gather(seed_i64, ov_opset.constant(1, Type.i32), ov_opset.constant(0, Type.i32)),
        increment
    ).output(0)

    key_tensor = ov_opset.gather(seed_i64, ov_opset.constant(0, Type.i32), ov_opset.constant(0, Type.i32)).output(0)

    key_tensor = ov_opset.reshape(key_tensor, ov_opset.constant([1], Type.i32), False).output(0)
    new_counter = ov_opset.reshape(new_counter, ov_opset.constant([1], Type.i32), False).output(0)

    new_seed = ov_opset.concat([key_tensor, new_counter], 0).output(0)
    if seed.get_element_type() != Type.i64:
        new_seed = ov_opset.convert(new_seed, seed.get_element_type()).output(0)

    return random_values, new_seed


def _random_normal(shape, dtype, seed):
    # Generates Normal distribution using Box-Muller transform
    dtype = dtype or floatx()
    target_type = OPENVINO_DTYPES[dtype]

    u1, seed = _stateless_uniform(shape, seed, dtype)
    u2, seed = _stateless_uniform(shape, seed, dtype)

    # Box-Muller
    # z0 = sqrt(-2 ln u1) * cos(2 pi u2)
    epsilon = ov_opset.constant(1e-7, target_type).output(0)
    u1 = ov_opset.add(u1, epsilon).output(0)

    const_two = ov_opset.constant(2.0, target_type).output(0)
    const_minus_two = ov_opset.constant(-2.0, target_type).output(0)
    const_two_pi = ov_opset.constant(2.0 * np.pi, target_type).output(0)

    ln_u1 = ov_opset.log(u1).output(0)
    sqrt_term = ov_opset.sqrt(ov_opset.multiply(const_minus_two, ln_u1)).output(0)

    angle = ov_opset.multiply(const_two_pi, u2).output(0)
    z0 = ov_opset.multiply(sqrt_term, ov_opset.cos(angle)).output(0)

    return z0, seed


def _random_gamma(shape, alpha, dtype, seed):
    dtype = dtype or floatx()
    target_type = OPENVINO_DTYPES[dtype]

    alpha = get_ov_output(alpha)
    if alpha.get_element_type() != target_type:
        alpha = ov_opset.convert(alpha, target_type).output(0)

    shape_tensor = get_ov_output(shape, Type.i32)
    if not isinstance(shape, (list, tuple)):
         shape_tensor = ov_opset.convert(shape_tensor, Type.i32).output(0)

    # Broadcast alpha to shape
    alpha = ov_opset.broadcast(alpha, shape_tensor).output(0)

    # Handle alpha < 1
    one_float = ov_opset.constant(1.0, target_type).output(0)
    is_less_than_one = ov_opset.less(alpha, one_float).output(0)

    # Boost alpha to be >= 1
    alpha_boosted = ov_opset.select(is_less_than_one, ov_opset.add(alpha, one_float), alpha).output(0)

    d = ov_opset.subtract(alpha_boosted, ov_opset.constant(1.0/3.0, target_type).output(0)).output(0)
    c = ov_opset.divide(one_float, ov_opset.sqrt(ov_opset.multiply(ov_opset.constant(9.0, target_type), d))).output(0)

    # Initialize loop variables
    # results: zeros
    # mask: all false (not done)
    # seed: seed

    results_init = ov_opset.broadcast(ov_opset.constant(0.0, target_type).output(0), shape_tensor).output(0)
    mask_init = ov_opset.broadcast(ov_opset.constant(False, Type.boolean).output(0), shape_tensor).output(0)

    # Loop condition: not all done
    def cond(results, mask, seed, c, d, one_float, shape_tensor):
        # We need to reduce mask to scalar boolean (True if ANY is False -> keep looping)
        # mask is True for Done. So we loop while NOT all(mask).

        # Flatten mask to make reduction easier or just reduce over all axes
        # We need to compute rank to reduce all axes?
        # Or reshape to 1D then reduce_logical_and.

        # To avoid dynamic rank issues, we can just use reshape([-1])
        mask_flat = ov_opset.reshape(mask, ov_opset.constant([-1], Type.i32).output(0), False).output(0)
        all_done = ov_opset.reduce_logical_and(mask_flat, ov_opset.constant(0, Type.i32).output(0), keep_dims=False).output(0)
        return ov_opset.logical_not(all_done).output(0)

    def body(results, mask, seed, c, d, one_float, shape_tensor):
        results = get_ov_output(results)
        mask = get_ov_output(mask)
        seed = get_ov_output(seed)
        c = get_ov_output(c)
        d = get_ov_output(d)
        one_float = get_ov_output(one_float)
        shape_tensor = get_ov_output(shape_tensor)

        # Generate candidates
        # Use shape_tensor passed into loop
        z, seed = _random_normal(shape_tensor, dtype, seed)
        u, seed = _stateless_uniform(shape_tensor, seed, dtype)

        v_base = ov_opset.add(one_float, ov_opset.multiply(c, z)).output(0)
        v = ov_opset.power(v_base, ov_opset.constant(3.0, target_type)).output(0)

        # Valid if v > 0
        v_pos = ov_opset.greater(v, ov_opset.constant(0.0, target_type)).output(0)

        z_sq = ov_opset.multiply(z, z).output(0)

        # Condition 1: u < 1 - 0.0331 * z^4
        z_pow4 = ov_opset.multiply(z_sq, z_sq).output(0)
        cond1_rhs = ov_opset.subtract(one_float, ov_opset.multiply(ov_opset.constant(0.0331, target_type), z_pow4)).output(0)
        accept1 = ov_opset.less(u, cond1_rhs).output(0)

        # Condition 2: log(u) < 0.5 * z^2 + d * (1 - v + log(v))
        # Computation for condition 2 is only safe if v > 0 (log(v)).
        # We can use select to avoid invalid log input or NaN propagation, though select executes both branches?
        # OpenVINO select executes both. We need to handle v <= 0 case to avoid log(<=0) -> NaN.
        safe_v = ov_opset.select(v_pos, v, one_float).output(0) # v=1 -> log(v)=0

        log_v = ov_opset.log(safe_v).output(0)
        log_u = ov_opset.log(u).output(0)

        term = ov_opset.add(ov_opset.subtract(one_float, v), log_v).output(0)
        cond2_rhs = ov_opset.add(
            ov_opset.multiply(ov_opset.constant(0.5, target_type), z_sq),
            ov_opset.multiply(d, term)
        ).output(0)

        accept2 = ov_opset.less(log_u, cond2_rhs).output(0)

        accepted = ov_opset.logical_or(accept1, accept2).output(0)

        # Must be valid (v > 0) AND accepted
        accepted = ov_opset.logical_and(v_pos, accepted).output(0)

        # Update results where: accepted AND NOT(mask) (i.e. not previously done)
        update_mask = ov_opset.logical_and(accepted, ov_opset.logical_not(mask)).output(0)

        new_val = ov_opset.multiply(d, v).output(0)

        new_results = ov_opset.select(update_mask, new_val, results).output(0)
        new_mask = ov_opset.logical_or(mask, accepted).output(0)

        return new_results, new_mask, seed, c, d, one_float, shape_tensor

    loop_vars = [results_init, mask_init, seed, c, d, one_float, shape_tensor]
    results, _, seed, _, _, _, _ = while_loop(cond, body, loop_vars)

    results = get_ov_output(results)

    # Correction for alpha < 1
    # X = X * U^(1/alpha)
    u2, seed = _stateless_uniform(shape, seed, dtype)

    power_exponent = ov_opset.divide(one_float, alpha).output(0)
    correction = ov_opset.power(u2, power_exponent).output(0)

    final_result = ov_opset.select(is_less_than_one, ov_opset.multiply(results, correction), results).output(0)

    # seed is technically consumed, but we don't return it from gamma as per signature
    # But if we were to chain, we'd need it.

    return final_result


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
    raise NotImplementedError(
        "`binomial` is not supported with openvino backend"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed_1 = draw_seed(seed)
    seed_1_ov = get_ov_output(seed_1)

    # We use two seeds for independent gamma samples
    offset = ov_opset.constant(np.array([12345, 67890]), dtype=Type.i32).output(0)
    seed_dtype = seed_1_ov.get_element_type()
    if seed_dtype != Type.i32:
        offset = ov_opset.convert(offset, seed_dtype).output(0)

    # Use multiplication for better mixing
    multiplier = ov_opset.constant(np.array([3, 5]), dtype=Type.i32).output(0)
    if seed_dtype != Type.i32:
        multiplier = ov_opset.convert(multiplier, seed_dtype).output(0)

    seed_2 = ov_opset.add(seed_1_ov, offset).output(0)
    seed_2 = ov_opset.multiply(seed_2, multiplier).output(0)

    gamma_a = _random_gamma(shape, alpha, dtype, seed_1_ov)
    gamma_b = _random_gamma(shape, beta, dtype, seed_2)

    sum_gamma = ov_opset.add(gamma_a, gamma_b).output(0)
    sample = ov_opset.divide(gamma_a, sum_gamma).output(0)
    return OpenVINOKerasTensor(sample)
