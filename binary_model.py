import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "PATH_TO_DATASET"  # update this
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32
EPOCHS = 20

MODEL_HANDLE = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"

# -----------------------------
# Dataset
# -----------------------------
def build_dataset(subset):
    return image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset=subset,
        seed=123,
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

train_ds = build_dataset("training")
val_ds = build_dataset("validation")

train_size = train_ds.cardinality().numpy()
val_size = val_ds.cardinality().numpy()

# Normalize images
normalization = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y)).repeat()
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# -----------------------------
# Model
# -----------------------------
feature_extractor = hub.KerasLayer(MODEL_HANDLE, trainable=False)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    feature_extractor,
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Training
# -----------------------------
steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# -----------------------------
# Save model
# -----------------------------
model.save("saved_model")

# -----------------------------
# Evaluation on independent dataset
# -----------------------------
ADDITIONAL_DATA_DIR = "PATH_TO_EXTERNAL_DATASET"  # update this

test_ds = image_dataset_from_directory(
    ADDITIONAL_DATA_DIR,
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False
)

# Normalize using same preprocessing
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

file_paths = test_ds.file_paths

correct = 0
total = 0
misclassified = []

for i, (img, true_label) in enumerate(test_ds):
    pred = model.predict(img, verbose=0)[0][0]
    pred_label = int(pred >= 0.5)
    true_label = int(true_label.numpy()[0])

    if pred_label == true_label:
        correct += 1
    else:
        misclassified.append((img[0].numpy(), true_label, pred_label, file_paths[i]))

    total += 1

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")

# Optional: inspect misclassified samples
class_names = ["neg", "pos"]

for img, true_label, pred_label, path in misclassified:
    plt.imshow(np.clip(img, 0, 1))
    plt.axis("off")
    plt.show()

    print(f"File: {path}")
    print(f"True: {class_names[true_label]} | Predicted: {class_names[pred_label]}\n")