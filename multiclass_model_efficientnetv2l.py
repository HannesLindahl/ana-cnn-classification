import os
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------
# Configuration
# -----------------------
DATA_DIR = "path/to/dataset"  # <-- change this
IMG_SIZE = (480, 480)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

USE_AUGMENTATION = False
USE_FINE_TUNING = False

# -----------------------
# Data loading
# -----------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# -----------------------
# Optional augmentation
# -----------------------
if USE_AUGMENTATION:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------
# Model
# -----------------------
base_model = tf.keras.applications.EfficientNetV2L(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = USE_FINE_TUNING

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------
# Callbacks
# -----------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )
]

# -----------------------
# Training
# -----------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------
# Save model
# -----------------------
os.makedirs("models", exist_ok=True)
model.save("models/efficientnetv2l_multiclass.keras")

# -----------------------
# Evaluation on external dataset (optional)
# -----------------------
def evaluate_on_directory(model, data_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int"
    ).prefetch(tf.data.AUTOTUNE)

    preds = model.predict(ds)
    pred_labels = preds.argmax(axis=1)
    true_labels = tf.concat([y for x, y in ds], axis=0).numpy()

    accuracy = (pred_labels == true_labels).mean()
    print(f"External dataset accuracy: {accuracy:.4f}")

# Example usage:
# evaluate_on_directory(model, "path/to/external_dataset")