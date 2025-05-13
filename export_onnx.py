import tensorflow as tf
import tf2onnx
import shutil

# 1. Load your trained Keras model
model = tf.keras.models.load_model("mlp_model.keras")

# 2. Monkey-patch output_names so tf2onnx.find_duplicate_names() won’t crash
#    We pull the op names from the model’s outputs.
try:
    model.output_names = [out.name.split(':')[0] for out in model.outputs]
except Exception:
    # If that fails for any reason, fall back to a single generic name
    model.output_names = ["output"]

# 3. Clean up any previous saved_model dir (not strictly needed here, but good hygiene)
saved_model_dir = "tmp_saved_model"
shutil.rmtree(saved_model_dir, ignore_errors=True)

# 4. Convert directly from Keras
#    This bypasses the CLI and the set_learning_phase call
onnx_model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec([None, model.input_shape[1]], tf.float32, name="input")],
    opset=13,
    output_path="mlp_model.onnx"
)