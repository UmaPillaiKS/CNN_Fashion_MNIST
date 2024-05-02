#Submitted by Uma Krishna Pillai, fdai8005

import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
import os
import sys

class DataPreprocessor:
    def __init__(self, data_path):
        self.loaded_data = np.load(data_path)
        self.image_data = self.loaded_data['image_data']
        self.image_labels = self.loaded_data['image_labels']
        #self.image_filenames = self.loaded_data['image_filenames']
        self.num_samples_to_visualize = 5
        self.random_indices = np.random.choice(len(self.image_data), self.num_samples_to_visualize, replace=False)

    def visualize_class_distribution(self):
        numeric_labels = [int(label) for label in self.image_labels]
        unique_labels, counts = np.unique(numeric_labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique_labels, counts, edgecolor='black')
        plt.title("Class Distribution")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.show()

    def compute_confusion_matrix(self, y_true, y_pred):
        num_classes = len(np.unique(y_true))
        confusion_matrix = np.bincount(num_classes * y_true + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)
        return confusion_matrix

    def visualize_confusion_matrix(self, confusion_matrix):
        plt.imshow(confusion_matrix)
        plt.show()

    def print_confusion_matrix(self, confusion_matrix):
        print("Confusion Matrix:")
        print(confusion_matrix)

    def preprocess_data(self):
        unique_classes = np.unique(np.argmax(self.image_labels, axis=1))
        print("Unique Classes:", unique_classes)

        resampled_labels = []
        resampled_images = []
        target_samples_per_class = 6000

        for class_label in unique_classes:
            class_indices = np.where(np.argmax(self.image_labels, axis=1) == class_label)[0]
            class_images = self.image_data[class_indices]
            class_labels = self.image_labels[class_indices]

            X_COUNT = class_images.shape[0]
            random_indices = np.random.randint(0, X_COUNT, size=target_samples_per_class)

            resampled_images.append(class_images[random_indices])
            resampled_labels.append(class_labels[random_indices])

        resampled_images = np.concatenate(resampled_images)
        resampled_labels = np.concatenate(resampled_labels)

        shuffle_indices = np.random.permutation(len(resampled_labels))
        resampled_images = resampled_images[shuffle_indices]
        resampled_labels = resampled_labels[shuffle_indices]

        
        
        normalized_images = resampled_images / resampled_images.max(axis=1,keepdims=True)
        num_samples = len(normalized_images)
        split_ratio = 0.8
        split_index = int(num_samples * split_ratio)

        return normalized_images, resampled_labels, split_index


def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, epochs, batch_size, validation_split):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_acc}')
    print(f'Test Loss: {test_loss}')


def main():
    if len(sys.argv) != 6:
        print("Usage: python3 cnn.py <npz> imgW imgH imgC train|test")
        sys.exit(1)

    data_path = sys.argv[1]
    imgW, imgH, imgC = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    train_test = sys.argv[5].lower()

    data_preprocessor = DataPreprocessor(data_path)

    if train_test == 'train':
        normalized_images, resampled_labels, split_index = data_preprocessor.preprocess_data()

        total_samples = len(normalized_images)
        print("Total Number of Samples in the Original Dataset:", total_samples)

        X_train, X_test = normalized_images[:split_index], normalized_images[split_index:]
        y_train, y_test = resampled_labels[:split_index], resampled_labels[split_index:]

        print("X_train Shape:", X_train.shape)
        print("X_test Shape:", X_test.shape)

        input_shape = (imgW, imgH, imgC)
        model = build_cnn_model(input_shape)

        epochs = 10
        batch_size = 32
        validation_split = 0.2

        history = train_model(model, X_train, y_train, epochs, batch_size, validation_split)
        model.save_weights('your_model_weights.h5')

        #data_preprocessor.plot_loss_curves(history)
        evaluate_model(model, X_test, y_test)

        # Get predictions on test set
        y_pred = np.argmax(model.predict(X_test), axis=1)

        # Compute and visualize confusion matrix
        confusion_matrix = data_preprocessor.compute_confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        data_preprocessor.visualize_confusion_matrix(confusion_matrix)
        data_preprocessor.print_confusion_matrix(confusion_matrix)

    elif train_test == 'test':
        # Load pre-trained model
        input_shape = (imgW, imgH, imgC)
        model = build_cnn_model(input_shape)
        model.load_weights('your_model_weights.h5')  # Change this to the path of your saved model weights

        # Load test data
        normalized_images, resampled_labels, split_index = data_preprocessor.preprocess_data()
        X_test, y_test = normalized_images[split_index:], resampled_labels[split_index:]

        evaluate_model(model, X_test, y_test)

        # Get predictions on test set
        y_pred = np.argmax(model.predict(X_test), axis=1)

        # Compute and visualize confusion matrix
        confusion_matrix = data_preprocessor.compute_confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        data_preprocessor.visualize_confusion_matrix(confusion_matrix)
        data_preprocessor.print_confusion_matrix(confusion_matrix)

    else:
        print("Invalid argument. Use 'train' or 'test'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
