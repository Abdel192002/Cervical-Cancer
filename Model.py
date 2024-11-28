from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalMaxPool2D, BatchNormalization, Dropout, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf

class ResNet50C:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5, learning_rate=1e-4):
       
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):
      
        # Charger le modèle de base ResNet50 préentraîné sur ImageNet
        r50 = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Geler les 50 premières couches
        for layer in r50.layers[:50]:
            layer.trainable = False
        # Déverrouiller les couches restantes
        for layer in r50.layers[50:]:
            layer.trainable = True

        # Ajouter des couches supplémentaires pour la classification
        x = r50.output
        x = GlobalMaxPool2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        # Créer le modèle complet
        self.model = Model(inputs=r50.input, outputs=output)

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
