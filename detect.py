import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import preprocess_input

# Define or import your custom layers here
# For example, if FixedDropout is a custom layer in your model
# from path.to.your.custom_module import FixedDropout

# Register the custom object
# custom_objects = {'FixedDropout': FixedDropout}

# Load the saved model with custom objects
# model = load_model('crop_disease_detection_model.h5', custom_objects=custom_objects)
model = load_model('sugarcane-disease-detector-colab.h5')

# Define the image path you want to predict
image_path = 'sugarcane-disease-dataset/train/yellow leaf/yldt99_jpg.rf.44c2000ec0bcaf12e79fab204774ec61.jpg'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Assuming you have a list of class labels, replace this with your specific labels
class_labels = [
    'cercospola',
    'eyespot',
    'healthy',
    'redrot',
    'wheat rust',
    'yellow leaf'
    ]

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]

# if healthy

if predicted_class_label=="healthy":
    print(predicted_class_label, int(round(predictions[0,2]*100)))
    print(predictions)

elif predicted_class_label=="cercospola":
    print(predicted_class_label, int(round(predictions[0,0]*100)))
    print(predictions)
    

elif predicted_class_label=="eyespot":
    print(predicted_class_label, int(round(predictions[0,1]*100)))
    print(predictions)

elif predicted_class_label=="redrot":
    print(predicted_class_label, int(round(predictions[0,3]*100)))
    print(predictions)

elif predicted_class_label=="wheat rust":
    print(predicted_class_label, int(round(predictions[0,4]*100)))
    print(predictions)

elif predicted_class_label=="yellow leaf":
    print(predicted_class_label, int(round(predictions[0,5]*100)))
    print(predictions)