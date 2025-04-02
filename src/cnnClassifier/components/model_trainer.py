import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None  # Initialize model variable

    def get_base_model(self):
        """Load the base model from the given path."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # Reinitialize the optimizer to avoid unknown variable issues
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Compile the model again to bind the new optimizer
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        """Prepare training and validation data generators."""
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model."""
        model.save(path)

    def train(self):
        """Train the model and handle debugging-related issues."""
        if self.model is None:
            raise ValueError("Base model is not loaded. Call get_base_model() first.")

        # Ensure eager execution is set properly
        tf.config.run_functions_eagerly(True)

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        try:
            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator
            )

            # Save the trained model
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )

            print(self.model.summary())

        except Exception as e:
            print(f"Training failed: {e}")

        # Enable debug mode for TensorFlow datasets
        tf.data.experimental.enable_debug_mode()
