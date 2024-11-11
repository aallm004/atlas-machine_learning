#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
IMG_SIZE = 331  # Increased image size
BATCH_SIZE = 16  # Smaller batch size for better generalization
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 20

# Load the ResNet50V2 model (better performance than VGG16)
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Create more sophisticated data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    validation_split=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

# Load data with increased image size
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

# Build enhanced model architecture
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)

# Add more sophisticated layers
x = BatchNormalization()(x)
x = Dense(1536, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(768, activation='relu')(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Dense(384, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Freeze base model
base_model.trainable = False

# Create callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        min_delta=1e-4
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=4,
        min_delta=1e-4,
        min_lr=1e-8
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Initial training with higher learning rate
print("Initial training phase...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# Fine-tuning phase
print("Fine-tuning phase...")
base_model.trainable = True

# Freeze first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# Final evaluation
print("Final evaluation...")
test_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

results = model.evaluate(test_generator)
print(f'Test Loss: {results[0]:.4f}')
print(f'Test Accuracy: {results[1]*100:.2f}%')
print(f'Test AUC: {results[2]:.4f}')

# Plot comprehensive training history
def plot_training_history(history, fine_tune_history=None):
    metrics = ['accuracy', 'loss', 'auc']
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(history.history[metric], label=f'Training {metric.title()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.title()}')
        
        if fine_tune_history:
            offset = len(history.history[metric])
            plt.plot(
                range(offset, offset + len(fine_tune_history.history[metric])),
                fine_tune_history.history[metric],
                label=f'Fine-tune Training {metric.title()}'
            )
            plt.plot(
                range(offset, offset + len(fine_tune_history.history[f'val_{metric}'])),
                fine_tune_history.history[f'val_{metric}'],
                label=f'Fine-tune Validation {metric.title()}'
            )
            
        plt.title(f'Model {metric.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history, history_fine)

# Save the final model
model.save('final_model.h5')


