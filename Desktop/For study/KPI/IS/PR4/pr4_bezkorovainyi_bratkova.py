import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import glob

mode = 0  # 0 - Shoe vs Sandal vs Boot Dataset, 1 - Chinese MNIST

# Вказуємо шлях до даних
if mode == 0:
    base_dir = "./Shoe vs Sandal vs Boot Dataset"
    train_dir = base_dir

    # Аугментація та розділення на train/val
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode="categorical",
        subset="training",
        shuffle=False,
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = train_generator.num_classes

elif mode == 1:
    csv_path = "./Chinese MNIST/chinese_mnist.csv"
    image_dir = "./Chinese MNIST/data"

    # Завантаження CSV
    df = pd.read_csv(csv_path)

    df["filename"] = df.apply(
        lambda row: f"data/input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg",
        axis=1,
    )
    df["filepath"] = df["filename"].apply(lambda f: os.path.join(image_dir, f))
    df = df[df["filepath"].apply(os.path.exists)]

    df["label"] = (df["value"] - 1).astype(str)

    # Ділимо на train/val
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Генератор зображень
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=(64, 64),
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
    )

    val_generator = datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label",
        target_size=(64, 64),
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
    )

    num_classes = len(train_generator.class_indices)

# Побудова моделі CNN

# model = Sequential(
#     [
#         Conv2D(
#             32,
#             kernel_size=(5, 5), # (3, 3)
#             strides=(2, 2), # (1, 1)
#             padding="same", # valid
#             activation="relu",
#             input_shape=(64, 64, 3),
#         ),
#         MaxPooling2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(128, activation="relu"),
#         Dropout(0.5),
#         Dense(num_classes, activation="softmax"),
#     ]
# )

model = Sequential(
    [
        Conv2D(16, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Навчання моделі
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stop],
)

# Оцінка моделі
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy:.4f}")


# Збереження історії навчання
pd.DataFrame(history.history).to_csv("training_log.csv", index=False)

# Візуалізація навчання
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Передбачення для валідаційної множини
val_preds = model.predict(val_generator)
val_pred_labels = np.argmax(val_preds, axis=1)
true_labels = val_generator.classes

# classification report
print("\nCNN Classification Report:\n")
print(classification_report(true_labels, val_pred_labels))

# AUC (One-vs-Rest)
y_true_bin = label_binarize(true_labels, classes=np.arange(num_classes))
auc = roc_auc_score(y_true_bin, val_preds, average="macro", multi_class="ovr")
print(f"CNN AUC: {auc:.4f}")


# MLP ================================================

if mode == 0:

    base_dir = "./Shoe vs Sandal vs Boot Dataset"

    # Отримуємо список усіх зображень і їх мітки
    categories = sorted(os.listdir(base_dir))
    X = []
    y = []

    for label_index, category in enumerate(sorted(categories)):
        category_dir = os.path.join(base_dir, category)
        image_files = glob.glob(os.path.join(category_dir, "*.jpg"))

        for filepath in image_files:
            img = load_img(filepath, target_size=(64, 64))
            img_array = img_to_array(img) / 255.0
            X.append(img_array.flatten())  # Розгортаємо зображення в вектор
            y.append(label_index)

    X = np.array(X)
    y = np.array(y)

    # Розділення на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Створення MLP з 3 шарами: 128, 128, 64
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=300,
        momentum=0.9,
        verbose=False,
    )

    # Навчання
    mlp.fit(X_train, y_train)

    # Оцінка
    y_pred = mlp.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, y_pred)

    print("\nMLP Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # AUC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_proba = mlp.predict_proba(X_test)
    mlp_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    print(f"MLP AUC: {mlp_auc:.4f}")

    print(f"\n\nMLP Accuracy: {mlp_accuracy:.4f}")
    print(f"\nCNN Accuracy: {accuracy:.4f}\n")

elif mode == 1:

    # Завантажуємо дані так само, як у CNN
    csv_path = "./Chinese MNIST/chinese_mnist.csv"
    image_dir = "./Chinese MNIST/data"
    df = pd.read_csv(csv_path)

    df["filename"] = df.apply(
        lambda row: f"data/input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg",
        axis=1,
    )
    df["filepath"] = df["filename"].apply(lambda f: os.path.join(image_dir, f))
    df = df[df["filepath"].apply(os.path.exists)]

    df["label"] = df["value"] - 1  # 0-9

    # Завантажуємо зображення та мітки
    X = []
    y = []

    for i, row in df.iterrows():
        img = load_img(row["filepath"], target_size=(64, 64))
        img_array = img_to_array(img) / 255.0
        X.append(img_array.flatten())  # Розгортання зображення у вектор
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    # Розбиваємо на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Створюємо та тренуємо MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 128, 64),
        activation="relu",
        solver="sgd",
        max_iter=300,
        verbose=False,
        learning_rate_init=0.4,
    )
    mlp.fit(X_train, y_train)

    # Оцінка моделі
    y_pred = mlp.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, y_pred)

    print("\nMLP Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # AUC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_proba = mlp.predict_proba(X_test)
    mlp_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    print(f"MLP AUC: {mlp_auc:.4f}")

    print(f"\n\nMLP Accuracy: {mlp_accuracy:.4f}")
    print(f"\n\nCNN Accuracy: {accuracy:.4f}\n")


#
if mode == 0:
    image_path = "./Shoe vs Sandal vs Boot Dataset/boot/boot (1).jpg"

    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0

    img_cnn = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

    cnn_prediction = model.predict(img_cnn)
    cnn_class = np.argmax(cnn_prediction)

    print(f"\nCNN передбачений клас: {cnn_class}\n")
