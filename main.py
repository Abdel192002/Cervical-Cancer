from DataLoaders import DataLoader
# Initialiser l'objet avec le chemin vers les données
import keras
print(keras.__version__)
data_loader = DataLoader(path = "C:/Users/nessa/Downloads/dataset_split_balanced/")

# Récupérer les ensembles
datasets = data_loader.get_datasets()

# Accéder aux différents ensembles
train_with_aug = datasets["train_with_augmentation"]
train_without_aug = datasets["train_without_augmentation"]
validation_set = datasets["validation"]
test_set = datasets["test"]


