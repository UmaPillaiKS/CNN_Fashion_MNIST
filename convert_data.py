#Submitted by Uma Krishna Pillai, fdai8005
from PIL import Image
import numpy as np
import os
import sys

class DataPreprocessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_data = []
        self.image_labels = []
        self.image_filenames = []

    def load_and_process_images(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".png"):
                file_path = os.path.join(self.folder_path, filename)
                
                label = int(filename.split('-')[0])
                one_hot_label = np.zeros(10)
                one_hot_label[label] = 1
                
                actual_filename = filename.split('-')[1]
                im_frame = Image.open(file_path)
                np_frame = np.array(im_frame)
                self.image_data.append(np_frame)
                self.image_labels.append(one_hot_label)
                self.image_filenames.append(actual_filename)

        self.image_data = np.array(self.image_data)
        self.image_labels = np.array(self.image_labels)
        self.image_filenames = np.array(self.image_filenames)

        np.savez('fashion_mnist_data.npz', image_data=self.image_data, image_labels=self.image_labels, image_filenames=self.image_filenames)

        # Determine the number of unique classes
        unique_classes = set(np.argmax(self.image_labels, axis=1))
        num_classes = len(unique_classes)

        # Print the number of classes and the unique classes
        print(f"There are {num_classes} classes in the dataset.")
        print(f"The unique classes are: {unique_classes}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 convert_data.py <path_to_images>")
        sys.exit(1)

    folder_path = sys.argv[1]
    data_preprocessor = DataPreprocessor(folder_path)
    data_preprocessor.load_and_process_images()
