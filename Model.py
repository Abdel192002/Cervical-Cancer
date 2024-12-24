import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, 
    GlobalMaxPool2D, BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class ResNet50C:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5, learning_rate=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):
        # Définir l'entrée
        input_image = Input(shape=self.input_shape, name="Image_Input")

        # Branche convolutionnelle pour les images
        x = Conv2D(32, (3, 3), activation="relu")(input_image)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)

        # Ajout d'une couche de classification
        output = Dense(self.num_classes, activation="softmax", name="Classification_Output")(x)

        # Construire le modèle
        self.model = Model(inputs=input_image, outputs=output)

        # Compiler le modèle
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model(self):
        if self.model is None:
            self.build_model()
        return self.model
