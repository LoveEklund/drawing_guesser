import tensorflow as tf
from tensorflow.keras import layers, models

def get_model(n_classes):
    """
    TODO: Maybe fix model to be something more advanced,
      but this worksgreat for now 
    """
    # Create the model
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the tensor output for dense layers
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(64, activation='relu'))

    # Output layer - Assume 10 classes for the sake of this example. Adjust this based on your specific dataset.
    model.add(layers.Dense(n_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model