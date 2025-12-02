#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================
# CONFIG
# ==============================================
AUTO = tf.data.AUTOTUNE

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Número CORRECTO para el dataset Kaggle Flowers:
NUM_CLASSES = 104

# Path local a los TFRecords en tu carpeta data/
DATA_PATH = "data/tfrecords-jpeg-224x224"

print("=== Entrenando Transfer Learning de Flores ===")
print("📦 Cargando TFRecords...")

# ==============================================
# TFRECORD PARSING
# ==============================================
def decode_example(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(example["class"], tf.int32)
    return image, label


def load_tfrecords(folder):
    files = tf.io.gfile.glob(f"{folder}/*.tfrec")
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    dataset = dataset.map(decode_example, num_parallel_calls=AUTO)
    return dataset


train_ds = load_tfrecords(f"{DATA_PATH}/train").shuffle(2048).batch(BATCH_SIZE).prefetch(AUTO)
val_ds   = load_tfrecords(f"{DATA_PATH}/val").batch(BATCH_SIZE).prefetch(AUTO)

# ==============================================
# TRANSFER LEARNING: MobileNetV2
# ==============================================
print("🧠 Construyendo MobileNetV2 Transfer Learning...")

base = keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  # congelamos MobileNet

inputs = keras.Input(shape=IMAGE_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================================
# TRAINING
# ==============================================
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
]

print("🚀 Entrenando modelo...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

print("=== ENTRENAMIENTO COMPLETADO ===")

# ==============================================
# OPCIONAL: Exportar a TFJS
# ==============================================
EXPORT_TFJS = True

if EXPORT_TFJS:
    print("💾 Exportando modelo a TensorFlow.js...")
    import tensorflowjs as tfjs
    os.makedirs("web_model_flowers", exist_ok=True)
    tfjs.converters.save_keras_model(model, "web_model_flowers")
    print("✅ Exportación completada: carpeta web_model_flowers/")
