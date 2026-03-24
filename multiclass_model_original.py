# Multiclass ANA pattern classifier (EfficientNetV2-XL)
# TensorFlow 2.15

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Paths (update as needed)
TRAIN_DIR = "/content/drive/MyDrive/ANA_cat_split"
TEST_DIR = "/content/drive/MyDrive/ANA_val_cat_split_1"
MODEL_SAVE_PATH = "/content/drive/MyDrive/saved_models/efficientnet_cat_lr001"

# Model configuration
MODEL_HANDLE = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32
NUM_CLASSES = 7

# Dataset loader
def build_dataset(data_dir, subset):
    return image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset=subset,
        seed=123,
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=1,
        shuffle=(subset == "training")
    )

train_ds = build_dataset(TRAIN_DIR, "training")
val_ds = build_dataset(TRAIN_DIR, "validation")

train_size = train_ds.cardinality().numpy()
val_size = val_ds.cardinality().numpy()

# Preprocessing
normalization = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.unbatch().batch(BATCH_SIZE)
train_ds = train_ds.map(lambda x, y: (normalization(x), y)).repeat()
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Class weights (from dataset distribution)
image_counts = {
    'cen': 191,
    'dfs': 58,
    'hom': 491,
    'kor': 470,
    'mem': 21,
    'nds': 55,
    'nuc': 265
}

total = sum(image_counts.values())
num_classes = len(image_counts)
class_weights = {
    i: total / (num_classes * count)
    for i, count in enumerate(image_counts.values())
}

# Model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODEL_HANDLE, trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES,
                          activation='softmax',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.build((None,) + IMAGE_SIZE + (3,))
model.summary()

# Training
steps_per_epoch = (train_size + BATCH_SIZE - 1) // BATCH_SIZE
validation_steps = (val_size + BATCH_SIZE - 1) // BATCH_SIZE

history = model.fit(
    train_ds,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    class_weight=class_weights
)

# Save model
model.save(MODEL_SAVE_PATH)

# -----------------------
# Evaluation on test data
# -----------------------

test_ds = image_dataset_from_directory(
    TEST_DIR,
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False
)

file_paths = test_ds.file_paths

correct = 0
total = 0
misclassified = []

class_names = ["cen", "dfs", "hom", "kor", "mem", "nds", "nuc"]

for idx, (img, label) in enumerate(test_ds):
    img = normalization(img)
    preds = model.predict(img, verbose=0)
    pred_label = np.argmax(preds)

    if pred_label == int(label[0]):
        correct += 1
    else:
        misclassified.append((img[0], label[0], pred_label, file_paths[idx]))

    total += 1

accuracy = correct / total
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Optional: visualize misclassifications
for img, true_label, pred_label, path in misclassified:
    plt.imshow(np.clip(img, 0, 1))
    plt.axis("off")
    plt.show()
    print(f"{path}")
    print(f"True: {class_names[int(true_label)]}, Pred: {class_names[pred_label]}")
