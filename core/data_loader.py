from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import config

def load_fer2013(data_dir=config.data_dir, batch_size=config.batch_size, validation_split=config.validation_split):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Train or test directory not found in {data_dir}")

    # Augmentation mạnh hơn cho tập train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Tăng góc xoay
        width_shift_range=0.3,  # Dịch chuyển nhiều hơn
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        shear_range=0.3,
        brightness_range=[0.7, 1.3],  # Thay đổi độ sáng
        validation_split=validation_split
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator