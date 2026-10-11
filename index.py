"""
Paper / Plastic / Empty image classifier - training script
-----------------------------------------------------------
Put your photos in folders like this (folder names = class names):

    dataset/
        empty/     <- photos of the empty platform
        paper/
        plastic/

Then run:
    python train.py

Output (in the "model" folder):
    segregator.keras   -> used by classify.py on your laptop
    labels.txt         -> class names, in order
    segregator_int8.tflite -> small version for ESP32-CAM later
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset", help="folder with one subfolder per class")
parser.add_argument("--out", default="model", help="where to save the trained model")
parser.add_argument("--size", type=int, default=96, help="image size (96 fits the ESP32-CAM)")
parser.add_argument("--epochs", type=int, default=20, help="training rounds (stage 1)")
parser.add_argument("--fine_tune_epochs", type=int, default=10, help="extra rounds (stage 2)")
parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width: 0.35 small/fast, 1.0 bigger/more accurate")
parser.add_argument("--no_pretrained", action="store_true", help="train from scratch (only for testing)")
args = parser.parse_args()

IMG = (args.size, args.size)
BATCH = 32
SEED = 42

# ---------- 1. Load the photos ----------
print(f"\nLoading photos from '{args.data}' ...")
train_ds = keras.utils.image_dataset_from_directory(
    args.data, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG, batch_size=BATCH, label_mode="int")
val_ds = keras.utils.image_dataset_from_directory(
    args.data, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG, batch_size=BATCH, label_mode="int")

class_names = train_ds.class_names
print("Classes found:", class_names)
for c in class_names:
    n = len([f for f in os.listdir(os.path.join(args.data, c)) if not f.startswith(".")])
    print(f"  {c}: {n} photos" + ("   <-- try to get at least 100" if n < 100 else ""))

# Balance classes so a class with more photos doesn't dominate
counts = np.zeros(len(class_names))
for _, y in train_ds.unbatch():
    counts[int(y)] += 1
class_weight = {i: counts.sum() / (len(counts) * max(c, 1)) for i, c in enumerate(counts)}

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------- 2. Build the model ----------
# Random changes to each photo while training, so the model doesn't just memorize
augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="augment")

base = keras.applications.MobileNetV2(
    input_shape=IMG + (3,), alpha=args.alpha, include_top=False,
    weights=None if args.no_pretrained else "imagenet")
base.trainable = False

# "core" = the model that gets saved and used for predictions (no augmentation inside)
core_in = keras.Input(shape=IMG + (3,))     # raw RGB pixels, 0-255
x = layers.Rescaling(1 / 127.5, offset=-1)(core_in)  # MobileNetV2 expects -1..1
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
core_out = layers.Dense(len(class_names), activation="softmax")(x)
core = keras.Model(core_in, core_out, name="segregator")

# Training model = augmentation + core
inputs = keras.Input(shape=IMG + (3,))
model = keras.Model(inputs, core(augment(inputs)))

callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                           restore_best_weights=True)]

# ---------- 3. Stage 1: train only the new last layer ----------
print("\nStage 1: training the classifier ...")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
          class_weight=class_weight, callbacks=callbacks)

# ---------- 4. Stage 2: fine-tune the top of MobileNetV2 ----------
if args.fine_tune_epochs > 0:
    print("\nStage 2: fine-tuning ...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
              class_weight=class_weight, callbacks=callbacks)

# ---------- 5. Check results ----------
print("\nChecking accuracy on the validation photos ...")
y_true, y_pred = [], []
for x_batch, y_batch in val_ds:
    p = core.predict(x_batch, verbose=0)
    y_true.extend(y_batch.numpy().tolist())
    y_pred.extend(np.argmax(p, axis=1).tolist())
y_true, y_pred = np.array(y_true), np.array(y_pred)
print(f"Validation accuracy: {np.mean(y_true == y_pred) * 100:.1f}%\n")

print("Confusion matrix (rows = real class, columns = predicted):")
print("            " + "".join(f"{c[:9]:>10}" for c in class_names))
for i, c in enumerate(class_names):
    row = [int(np.sum((y_true == i) & (y_pred == j))) for j in range(len(class_names))]
    print(f"{c[:10]:>10}  " + "".join(f"{v:>10}" for v in row))

# ---------- 6. Save ----------
os.makedirs(args.out, exist_ok=True)
keras_path = os.path.join(args.out, "segregator.keras")
core.save(keras_path)
with open(os.path.join(args.out, "labels.txt"), "w") as f:
    f.write("\n".join(class_names))
print(f"\nSaved {keras_path} and labels.txt")

# Small int8 version for the ESP32-CAM (optional)
try:
    saved_dir = os.path.join(args.out, "_saved_model")
    core.export(saved_dir, verbose=False)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative():
        for x_batch, _ in train_ds.take(10):
            for img in x_batch:
                yield [tf.cast(img[None, ...], tf.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite = converter.convert()
    tfl_path = os.path.join(args.out, "segregator_int8.tflite")
    with open(tfl_path, "wb") as f:
        f.write(tflite)
    print(f"Saved {tfl_path} ({len(tflite) / 1024:.0f} KB) for ESP32-CAM later")
except Exception as e:
    print(f"(Skipped ESP32 .tflite export: {str(e).splitlines()[0][:200]})")

print("\nDone! Next: run  python classify.py")