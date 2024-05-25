import tensorflow as tf
import numpy as np
import time

def cudnn_performance_test():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_train = np.random.random((10000, 28, 28, 1))
    y_train = np.random.randint(10, size=(10000,))

    start_time = time.time()
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=2)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"CuDNN performans testi tamamlandı! Eğitim süresi: {training_time:.2f} saniye")

cudnn_performance_test()
