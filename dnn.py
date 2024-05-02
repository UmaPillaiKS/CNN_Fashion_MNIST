import numpy as np
import requests
import os
import tensorflow as tf

class DNN:
    def __init__(self):
        # Initialize layers
        self.reshape_layer = tf.keras.layers.Reshape((28 * 28,), input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(100, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def __call__(self, inputs):
        # Define forward pass
        x = self.reshape_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

if __name__ == "__main__":
    # Download and load MNIST dataset
    if not os.path.exists("./mnist.npz"):
        print("Downloading MNIST...")
        fname = 'mnist.npz'
        url = 'http://www.gepperth.net/alexander/downloads/'
        r = requests.get(url + fname)
        open(fname, 'wb').write(r.content)

    # Load MNIST dataset
    data = np.load("mnist.npz")
    train_images, train_labels = data["arr_0"], data["arr_2"]
    test_images, test_labels = data["arr_1"], data["arr_3"]
    test_images = test_images.reshape(10000,28,28)

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Verify data shapes
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)

    # Reshape train_labels and test_labels
    train_labels = train_labels.argmax(axis=1)
    test_labels = test_labels.argmax(axis=1)

    # Create DNN model
    model = DNN()

    # Define and compile model
    inputs = tf.keras.Input(shape=(28, 28))
    outputs = model(inputs)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(train_images, train_labels, epochs=5, batch_size=32)

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')
