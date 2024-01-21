
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0  # You need to install efficientnet via pip install efficientnet

# Define the paths to your training and validation datasets
train_data_dir = 'sugarcane-disease-dataset/train'
validation_data_dir = 'sugarcane-disease-dataset/test'

# Set the image size according to the EfficientNet variant you choose
img_size = (224, 224)

# Create a data generator with data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create a data generator for the validation set
val_datagen = ImageDataGenerator(rescale=1./255)

# Define the batch size
batch_size = 70

# number of classess

num_classes = 6

# Create training set generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Change to 'binary' if you have two classes
)

# Create validation set generator
validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the EfficientNet model
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create your custom head for classification
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # num_classes is the number of classes in your dataset
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Started Training")

# Train the model
epochs = 10  # You may need to adjust this based on your dataset
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
model.save("Sugarcane-disease-detector-colab.h5")
print("Model Saved")