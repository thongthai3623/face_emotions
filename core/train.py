import logging
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import config
from data_loader import load_fer2013
from model import build_emotion_model
from utils import setup_logging
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(data_dir=config.data_dir):
    setup_logging()
    logging.info("Starting model training...")

    train_gen, val_gen, test_gen = load_fer2013(data_dir)
    logging.info("Data loaded successfully.")

    # Tính class weights
    classes = np.arange(config.num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_gen.classes)
    class_weight_dict = dict(enumerate(class_weights))

    model = build_emotion_model(config.input_shape, config.num_classes)
    logging.info("Model built successfully.")

    checkpoint = ModelCheckpoint(config.model_path, save_best_only=True, monitor='val_accuracy', save_format='h5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    callbacks = [checkpoint, early_stopping]

    logging.info(f"Training for {config.epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    test_loss, test_acc = model.evaluate(test_gen)
    logging.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # Vẽ đồ thị
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    return model, history

if __name__ == "__main__":
    train_model()