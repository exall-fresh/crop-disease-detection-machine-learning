CROP DISEASE DETECTION USING MACHINE LEARNING

Crop diseases can have a detrimental impact on agricultural yield. Early detection of these diseases is crucial for implementing timely interventions and preventing widespread damage. This project addresses the detection of sugarcane diseases using the state-of-the-art EfficientNet architecture, providing an efficient and accurate solution for farmers and agricultural practitioners.

1. INSTALLATION
To install the project in your command line write the command
"git clone https://github.com/exall-fresh/crop-disease-detection-machine-learning"
Then go into the project folder by using the command "cd crop-disease-detection-machine-learning"
2. INSTALLATION OF PACKAGES
To install the packages used in the project write the command:
"pip install -r requirements.txt"
3. DETECTION
To make disease detections on the project first in the detect.py file edit the image path on line number 18 as show below
"image_path = 'your_image_path/image_name'"
The run the detect.py file using the command
"py detect.py"
4. TRAIN YOUR CUSTOM MODEL
To train your own model create a dataset folder and inside the dataset folder create train and test folders, then in the train and test folders place the folders of the distinct classes of image that you are working on then run the script "py train.py" and the training will begin.
