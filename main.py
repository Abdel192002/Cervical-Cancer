from DataLoaders import DataLoader
from Model import ResNet50C
import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input

print(keras.__version__)
data_loader = DataLoader(path = "C:/Users/nessa/Downloads/dataset_split_balanced/")

# Récupérer les ensembles
datasets = data_loader.get_datasets()

# Accéder aux différents ensembles
train_with_aug = datasets["train_with_augmentation"]
train_without_aug = datasets["train_without_augmentation"]
validation_set = datasets["validation"]
test_set = datasets["test"]
r50 = ResNet50C(input_shape=(224, 224, 3), num_classes=5, learning_rate=1e-4)
model = r50.get_model()
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
   EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]
history = model.fit(
   train_with_aug,
    epochs=50,
    validation_data=validation_set, verbose=2,callbacks=callbacks
)

val_accuracy = history.history['val_accuracy'][-1]
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")