
import os

os.environ["KERAS_BACKEND"] = "openvino"

import keras
import numpy as np

def test_lstm_openvino():
    input_dim = 5
    units = 3
    batch_size = 2
    timesteps = 4

    inputs = np.random.random((batch_size, timesteps, input_dim)).astype("float32")

    # Use "zeros" or "glorot_uniform" for recurrent_initializer to avoid "qr" (orthogonal)
    lstm_layer = keras.layers.LSTM(
        units,
        use_cudnn="auto",
        recurrent_initializer="glorot_uniform",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros"
    )

    try:
        output = lstm_layer(inputs)
        print("LSTM Forward pass successful")
        print("Output shape:", output.shape)
    except NotImplementedError as e:
        print("Caught expected NotImplementedError:", e)
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lstm_openvino()
