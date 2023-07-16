# One-shot Facial Recognition
# Program utilizes VGGFace model to identify a single face in image (you'll need to include a picture of yourself, or whoever you want to
# be identified in the directory stated in the code).

# Utilizes one-shot learning, only one picture for recognition system

# CNN implementation based on the VGG-Very-Deep-16 CNN architecture
# Training data = Labeled Faces in the Wild and the YouTube Faces dataset.

# Place a few photos of people in the folder called ./person - or utilize the ones in there already
# Place an image of desired target into person folder as well.
# Faces are extracted using the haarcascade_frontface_default detector model and placed in the group of faces folder
# 5 extracted faces will be loaded and utilized for the one-shot learning model - political figures used- 

from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as pltdokc
from keras.models import model_from_json

# Load HAARCascade Face Detector
face_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Directory of images of people to extract faces from
mypath = "./person/"
image_file_names = [f for f in listdir(mypath) if isfile(join(mypath,f))]
print("collected image names")
print(image_file_names)

for image_name in image_file_names:
    person_image = cv2.imread(mypath + image_name)
    face_info = face_detector.detectMultiScale(person_image, 1.3, 5)
    for (x, y, w, h) in face_info:
        face = person_image[y:y+h, x:x+w]
        roi = cv2.resize(face, (128,128), interpolation= cv2.INTER_CUBIC)
    path = "./group_of_faces/" + "face_" + image_name
    cv2.imwrite(path, roi)
    cv2.imshow("face", roi)
    cv2.waitKey(2000)
cv2.destroyAllWindows()

# VGG model expects 224x224x3 sized input images - w the 3 representing the color channels (hsv, rgb, whatever).
# Images must be loaded from path and resized.
# Keras works with batches of images. When single image is loaded, the shape is (size1,size2,channels).
# In order to create a batch of images, you need an additional dimension: (batchsamples, size1,size2,channels)
# Preprocess_input function is meant to adjust imgs to the format the model requires.
def preprocess_image(image_path):
    # This PIL image instance
    img = load_img(image_path, target_size=(224,224))
    # Convert to numpy array
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Loads the VGGFace model
def loadVggFaceModel():
    model = Sequential()
    # First block
    # Adding zero on top, bottom, left and right
    model.add( ZeroPadding2D((1,1), input_shape=(224,224,3)) )
    # 64 3x3 filters with relu as the activation
    model.add( Convolution2D(64, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(64, (3,3), activation='relu') )
    model.add( MaxPooling2D((2,2), strides=(2,2)) )

    # Second block
    # 128 3x3 filters with relu as the activation
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(128, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(128, (3,3), activation='relu') )
    model.add( MaxPooling2D((2,2), strides=(2,2)) )

   
    # Third block
      # 256 3x3 filters with relu as the activation
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(256, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(256, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(256, (3, 3), activation='relu') )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)))

    
    # Fourth block
    # 512 3x3 filters with relu as the activation
    model.add( ZeroPadding2D((1, 1)) )
    model.add( Convolution2D(512, (3, 3), activation='relu') )
    model.add( ZeroPadding2D((1, 1)) )
    model.add( Convolution2D(512, (3, 3), activation='relu') )
    model.add( ZeroPadding2D((1, 1)) )
    model.add( Convolution2D(512, (3, 3), activation='relu') )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)) )

    # Fifth block
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(512, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(512, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(512, (3,3), activation='relu') )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)) )
    # After the convolutional layers, there are two fully connected layers with 4096 units each
    # Followed by dropout regularization to prevent overfitting.
    model.add( Convolution2D(4096, (7,7), activation='relu') )
    model.add( Dropout(0.5) )
    model.add( Convolution2D(4096, (1,1), activation='relu') )
    model.add( Dropout(0.5) )
    # Next line represents the number of unique identities in the training dataset
    model.add( Convolution2D(2622, (1,1)) )
    model.add( Flatten() )
    model.add( Activation('softmax') )

    model.load_weights('vgg_face_weights.h5')
    print(model.summary())

    # Use previous layer of the output layer for representation
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor


model = loadVggFaceModel()
print("Model Loaded")

# Vector Similarity
# Input images represented as vectors. Vector representations will be compared to determine
# If the target in the image is the same as the webcam

# Cosine distance is equal to 1 minus cosine similarity
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Test model using webcam
# Following lines look up the faces you extracted in the "group_of_faces" and uses the similarity Cosine
# Similarity func to detect which face is most similar to the one being extracted via your webcam

# Points to extracted faces
people_pictures = "./group_of_faces"

# Dictionary with its key as person name, value as vector representation
all_people_faces = dict()

for file in listdir(people_pictures):
    person_face, extension = file.split(".")
    img = preprocess_image('./group_of_faces/%s.jpg' % (person_face))
    # Represent images as 2622 dimensional vector 
    face_vector = model.predict(img)[0,:]
    all_people_faces[person_face] = face_vector
    print(person_face, face_vector)

print("Face representations retrived successfully")


# Following lines look up the faces extracted in the "group_of_faces" and uses similarity function
# To detect which faces are the most similar to the one being extacted via your webcam. Proper insertion of the target image in person directory
# Should result in your/the target's face being the most similar. 

cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()

    faces = face_detector.detectMultiScale(img, 1.3, 5)

    # Go through all faces
    for(x,y,w,h) in faces:
        # Adjust accordingly if your webcam resolution is higher
        if w > 100:
            # Draw a rectangle to face
            cv2.rectangle(img, (x,y), (w+h, y+h), (255,0,0), 2)

            # Crop detected face
            detected_face = img[y:y+h, x:x+w]
            # Resize to 224 x 224
            detected_face = cv2.resize(detected_face, (224,224))
            # Convert image to numpy array
            img_pixels = image.img_to_array(detected_face)
            # Expand its dimensionality for keras
            img_pixels = np.expand_dims(img_pixels, axis=0)
            # Normalize pixels between 0 and 1
            # Normalization will speed up training and avoid gradient exposion
            img_pixels /= 255

            # Pass the image to predictor in order to produce representation vector
            captured_representation = model.predict(img_pixels)[0,:]

            found = 0
            # Go through picture database (all people faces dict) to compare each face
            for i in all_people_faces:
                # This is the key of dictionary
                person_name = i
                # This is vector representation of faces of given name
                representation = all_people_faces[i]

                # Compare the detected face in all databases
                similarity = findCosineSimilarity(representation, captured_representation)

                # If we find the match, attach target's name to detected face
                if(similarity < 0.30):
                    cv2.putText(img, person_name[5:], (x+w+15, y-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    found = 1
                    break

            if(found == 0):
                cv2.putText(img, 'unknown', (x+w+15, y-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # End of all loops
    cv2.imshow('img', img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()