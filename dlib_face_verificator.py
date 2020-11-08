#Code example adapted from ....

import dlib
import numpy as np
import os
import time
# Get the face recognition model that produces encodings for the faces 
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
#win = dlib.image_window()


def get_embeddings(filename):
    '''
    get faces from given image using hog
    Perform face nomalosation 
    Generate the encodings 
    :param filename:
    :param required_size:
    :return:
    '''
    # Load image from file
    image = dlib.load_rgb_image(filename)
    #Create the detector
    face_detector = dlib.get_frontal_face_detector()
    # Detect faces 
    detected_faces = face_detector(image, 1)    
    # Get the Pose Predictor from dlib that produces 68 points for normalisation
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   
    # Get landmarks of those faces
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]    
    # For the face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]
    
    
def validate(ground_truth_file_path, candidate_file_path):
    '''
    Validate a candidate against an image using a given model
    :param ground_truth_file_path:
    :param candidate_file_path:
    :param model: defaults to Resnet50
    :return: score between 0 and 1 (lower is better)
    '''
    #candidate_embedding =[]
    #ground_truth_embedding =[]
    # get embeddings file filenames
    t1 = time.time()
    candidate_embedding = np.array((get_embeddings(candidate_file_path))[0])
    ground_truth_embedding = np.array((get_embeddings(ground_truth_file_path))[0])
    TOLERANCE =0.6
    prediction = (np.linalg.norm(candidate_embedding - ground_truth_embedding, axis=0) <= TOLERANCE)
    confidence = (np.linalg.norm(candidate_embedding - ground_truth_embedding, axis=0))
    runtime = time.time() - t1
    
    #dis = (np.linalg.norm(candidate_embedding - ground_truth_embedding, axis=0) <= TOLERANCE)
    return prediction,confidence,runtime



if __name__ == "__main__":
    
    test,test2,runtime = validate('mueez3.JPEG','Selena_Gomez1.jpg')
    print(test)
    print(test2)
    print("runtime: " + str(runtime))