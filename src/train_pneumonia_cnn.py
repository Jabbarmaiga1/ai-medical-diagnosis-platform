"""
Détection de Pneumonie sur Radiographies Thoraciques
Réseau de Neurones Convolutif (CNN) - TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

print("="*60)
print("🩻 DÉTECTION DE PNEUMONIE SUR RADIOGRAPHIES")
print("="*60)

# Vérifier le dataset
data_dir = 'data/chest_xray'
if not os.path.exists(data_dir):
    print(f"❌ Dataset non trouvé dans {data_dir}")
    print("📥 Télécharge-le depuis: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    exit()

# Paramètres
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

print("\n📊 Chargement des données...")

# Data augmentation pour éviter le sur-apprentissage
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Charger les datasets
train_generator = train_datagen.flow_from_directory(
    f'{data_dir}/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

val_generator = val_test_datagen.flow_from_directory(
    f'{data_dir}/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    f'{data_dir}/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print(f"\n✅ Classes: {train_generator.class_indices}")
print(f"✅ Train: {train_generator.samples} images")
print(f"✅ Validation: {val_generator.samples} images")
print(f"✅ Test: {test_generator.samples} images")

# ============================================================
# MODÈLE CNN (Transfer Learning avec ResNet50)
# ============================================================
print("\n🧠 Construction du modèle CNN...")

# Utiliser ResNet50 pré-entraîné (ImageNet)
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Geler les couches du base model (pas d'entraînement)
base_model.trainable = False

# Ajouter nos propres couches
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compiler
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss'),
    callbacks.ModelCheckpoint('models/pneumonia_cnn_best.h5', save_best_only=True, monitor='val_accuracy')
]

# ============================================================
# ENTRAÎNEMENT
# ============================================================
print("\n🚀 Début de l'entraînement...")
print("-" * 40)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1
)

# ============================================================
# ÉVALUATION
# ============================================================
print("\n📊 ÉVALUATION SUR LE TEST SET")
print("-" * 40)

test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(test_generator)

print(f"✅ Accuracy:  {test_acc:.4f}")
print(f"✅ AUC-ROC:   {test_auc:.4f}")
print(f"✅ Precision: {test_precision:.4f}")
print(f"✅ Recall:    {test_recall:.4f}")
print(f"✅ F1-Score:  {2 * test_precision * test_recall / (test_precision + test_recall):.4f}")

# ============================================================
# SAUVEGARDE
# ============================================================
model.save('models/pneumonia_cnn_model.h5')
print("\n✅ Modèle sauvegardé: models/pneumonia_cnn_model.h5")

# ============================================================
# GRAPHIQUES
# ============================================================
print("\n📈 Génération des graphiques...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('reports/pneumonia_training_history.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("🎉 MODÈLE PNEUMONIE ENTRAÎNÉ AVEC SUCCÈS !")
print(f"🏆 Accuracy sur test: {test_acc:.4f}")
print("="*60)