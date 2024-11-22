#!/usr/bin/env python3
from tensorflow import keras as K


class TrainingProgressCallback(K.callbacks.Callback):    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('accuracy', logs.get('categorical_accuracy', 0.0))
        val_acc = logs.get('val_accuracy', logs.get('val_categorical_accuracy', 0.0))
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print("-" * 50)

def preprocess_data(X, Y):
    """Preprocesses CIFAR-10 data with normalization"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    X_p = K.applications.resnet50.preprocess_input(
        K.backend.cast(X, 'float32')
    )
    return X_p, Y_p

if __name__ == '__main__':

    print("Loading and preprocessing data")
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Enhanced data augmentation
    data_augmentation = K.Sequential([
        K.layers.RandomFlip("horizontal"),
        K.layers.RandomRotation(0.15),
        K.layers.RandomZoom(0.1),
        K.layers.RandomTranslation(0.1, 0.1),
        K.layers.RandomBrightness(0.2),
        K.layers.RandomContrast(0.2)
    ])

    print("Building model...")
    base_model = K.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(64, 64, 3),
        pooling='avg'
    )
    
    # Progressive layer unfreezing
    for layer in base_model.layers:
        layer.trainable = False
    
    # Unfreeze last 30 layers except batch norm
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = True

    # Build model with improved architecture
    inputs = K.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = K.layers.UpSampling2D(size=(2, 2))(x)
    x = base_model(x)
    
    # Add more sophisticated head
    x = K.layers.Dense(1024, kernel_regularizer=K.regularizers.l2(1e-4))(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Dropout(0.4)(x)
    
    x = K.layers.Dense(512, kernel_regularizer=K.regularizers.l2(1e-4))(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Dropout(0.3)(x)
    
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)

    # Enable mixed precision with loss scaling
    policy = K.mixed_precision.Policy('mixed_float16')
    K.mixed_precision.set_global_policy(policy)
    
    # Use cosine decay learning rate
    initial_learning_rate = 1e-3
    decay_steps = 50 * len(X_train_p) // 64
    lr_schedule = K.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, alpha=1e-5
    )
    
    optimizer = K.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = K.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting training...")
    
    # Train with larger batch size and gradient accumulation
    history = model.fit(
        X_train_p,
        Y_train_p,
        batch_size=64,
        epochs=50,
        validation_data=(X_test_p, Y_test_p),
        callbacks=[
            TrainingProgressCallback(),
            K.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            K.callbacks.ModelCheckpoint(
                'cifar10_best.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ],
        verbose=1
    )

    model.save('cifar10.h5')
