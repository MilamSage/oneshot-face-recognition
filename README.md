
# One-shot Facial Recognition

Utilizing cosine similarity, program leverages discriminative features learned by the VGGFace model to compare and recognize faces accurately, even with limited training data. This implements one-shot learning. User is able to provide an image of a target in the appropriate directory to be identified on their webcam. 


## Context
This project is centralized around the VGGFace model, which is based on the VGG-Very-Deep-16 convolutional neural network (CNN) architecture. The model is pre-trained on large-scale face recognition datasets, including Labeled Faces in the Wild and the YouTube Faces dataset. This model has already been trained to extract discriminative features from images of faces.

## Methodology

1. **Face / Feature Extraction**: Faces are extracted from a set of images in the `person` directory using the Haarcascade Frontal Face Detector model. The extracted faces are saved in the `group_of_faces` directory.

2. **Data Preprocessing**: The faces extracted from the images are preprocessed to meet the input requirements of the VGGFace model. Each face image is resized to 224x224 pixels and normalized to ensure compatibility with the model. The image shape/dimension is further manipulated for utilization of keras batch samples.

3. **Feature Comparison (Cosine Similarity)**: The VGGFace model is loaded and layers of the convolutional network are established. Face images from the `group_of_faces` directory are input into the model to extract unique features and each respective image's features are stored as vector representations. These representations are run through a function designed to determine cosine similarity. This is ultimately how targets are "recognized" or identified.
  
4. **Real-time Recognition**: The system uses the webcam to capture video frames. Each frame is processed to detect faces using the same Haarcascade methodology previously mentioned -detected faces are resized, normalized, and input into the model to obtain face representations. During the recognition process, the system computes the cosine similarity between the face representation of the captured face from the webcam and the face representations of the known faces in the `group_of_faces` directory.
A lower cosine similarity score indicates a higher similarity between the two face representations. Threshold value of 0.30 is used to determine whether face detected via webcam is "identical" to known faces in `group_of_faces`.
If a match is found, the person's name associated with the matching face representation is displayed on the webcam feed. If no match is found, the face is labeled as "unknown".



## Dependencies
Following dependencies required.:
- Python 3
- OpenCV
- Tensorflow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib

Requirements.txt provided:
   `pip install -r requirements.txt`.
   
You will also need to download the pretrained weights for the model and place the .h5 file in the root directory:

[Download Them Here](https://www.dropbox.com/scl/fi/box3i9m3jlm60cmipcxis/vgg_face_weights.h5?rlkey=rp6uci2ngzpsd0cz9qhrmfwif&dl=00)


## Usage
Follow the steps below to use this facial recognition system:

1. Place an image of the person you want to identify in the `person` directory. Be sure to label the jpeg as whatever you want the target to be known as.
  eg. "bobby.jpg" for bobby

3. Add a few more images of different people in the `person` directory. Ensure these images only have one face present in them as the model is not capable of 
  handling extraction of multiple faces. Don't worry about image sizing - this program preprocesses everything.

5. Run the `facial_recognition.py` script.
6. The script will extract faces from the images in the `person` directory and save them in the `group_of_faces` directory.
7. The VGGFace model will be loaded, and the face representations of extracted faces will be computed.
8. The webcam will open, and the system will compare the faces detected by the webcam with the faces in the `group_of_faces` directory.
9. If a match is found, the person's name will be displayed on the webcam feed. Otherwise detected face will be labeled as "unknown"
10. Press the `Enter` key to exit the program.


