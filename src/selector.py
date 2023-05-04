import math
import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from . import utils


def get_all_faces_and_eyes_from_image(image, facial_landmark_predictor, debug):
    """
    This function accept an image and locate the eyes within it, computes eye coordinates and
    classifies each eye as open or closed.

    @param image: An input image.
    @param facial_landmark_predictor: A facial landmark predictor object that can detect facial landmarks in the image.
    @param debug: A boolean indicating whether to print debug information.
    :return: A list of tuples, where each tuple contains the facial landmarks, cropped image, cropped face landmarks,
    face embedding, and cropped eyes for a detected face.
    """
    faces = utils.FACE_DETECTOR(image, 1)
    print("number of faces in get_all_faces_and_eyes_from_image ", len(faces))

    # Crop to each face separately.
    results = []
    for face in faces:
        cropped_image = _get_cropped_image(image, face, padding_pct=15, debug=debug)
        cropped_face, cropped_face_landmarks, cropped_eyes = _get_eyes_from_image(
            cropped_image, facial_landmark_predictor)
        if len(cropped_eyes) != 2:
            print("Skipping bc too few eyes ", cropped_face_landmarks, cropped_eyes)
            continue
        face_embedding = face_recognition.face_encodings(cropped_image)[0]
        uncropped_landmarks = face_utils.shape_to_np(facial_landmark_predictor(image, face))
        results.append((uncropped_landmarks, cropped_image, face_embedding, cropped_eyes))

    return results


def _get_cropped_image(image, face_rect, padding_pct=50, debug=False):
    """
    This function takes an image and a rectangular bounding box representing a face in the image,
    and returns a cropped image of the face with some padding around the edges.

    @param image: The input image.
    @param face_rect: A rectangular bounding box representing the face to be cropped.
    @param padding_pct: The percentage of padding to add around the edges of the face.
    @param debug: A boolean indicating whether to print debug information.
    :return: A cropped image of the face with padding around the edges.
    """
    x, y, w, h = face_utils.rect_to_bb(face_rect)
    padding = math.floor((padding_pct / 100) * max(w, h))

    x_min = max(0, x - padding)
    x_max = min(x + w + padding, image.shape[1])
    y_min = max(0, y - padding)
    y_max = min(y + h + padding, image.shape[0])

    # Copy the array to a new image, as numpy <> dlib's interaction does not
    # work well with slices.
    cropped_image = image[y_min:y_max, x_min:x_max, :].copy()

    if debug:
        plt.imshow(image)
        plt.title("Original face")
        plt.show()
        plt.imshow(cropped_image)
        plt.title("Cropped face")
        plt.show()

        faces = utils.FACE_DETECTOR(cropped_image, 1)
        print("Faces detected in get_cropped_face cropped image version ", faces)
        copy = cropped_image.copy()
        for face in faces:
            x, y, w, h = face_utils.rect_to_bb(face)
            cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 0, 255), 2)

        plt.imshow(copy)
        plt.title("Boxes")
        plt.show()

    return cropped_image


def _get_eyes_from_image(image, facial_landmark_predictor):
    """
    This function accepts an image and locates the eyes within it. Computes eye coordinates and
    classifies each eye as open or closed.

    @param image: The input image.
    @param facial_landmark_predictor: A facial landmark predictor object that can detect facial landmarks in the image.
    :return: The face object, the face landmarks, and eye information.
    """
    faces = utils.FACE_DETECTOR(image, 1)
    face = None
    if len(faces) >= 1:
        face = faces[0]
    elif len(faces) == 0:
        print("No faces found in image")
        return None, None, ()

    face_landmarks = facial_landmark_predictor(image, face)
    face_landmarks = face_utils.shape_to_np(face_landmarks)
    left_eye = face_landmarks[utils.LEFT_EYE_START:utils.LEFT_EYE_END]
    right_eye = face_landmarks[utils.RIGHT_EYE_START:utils.RIGHT_EYE_END]
    left_ear = utils.eye_aspect_ratio(left_eye)
    right_ear = utils.eye_aspect_ratio(right_eye)

    eyes_coordinates, surrounding_coordinates = utils.get_eyes_and_surrounding_coordinates(face_landmarks)
    expanded_eyes_coordinates = utils.expand_eye_coordinates(eyes_coordinates, surrounding_coordinates)

    left_eye = {
        "kind": "left",
        "EAR": left_ear,
        "status": "closed" if left_ear < utils.MINIMUM_EAR else "open",
        "centroid": utils.centroid(left_eye),
        "coordinates": expanded_eyes_coordinates['left'],
    }

    right_eye = {
        "kind": "right",
        "EAR": right_ear,
        "status": "closed" if right_ear < utils.MINIMUM_EAR else "open",
        "centroid": utils.centroid(right_eye),
        "coordinates": expanded_eyes_coordinates['right'],
    }

    return face, face_landmarks, (left_eye, right_eye)


def compute_replacements(target_faces, eye_candidates):
    """
    This function takes a list of target faces and a list of eye candidates and computes the best replacement eyes
    for each target face by minimizing the Euclidean distance between face embeddings.

    @param target_faces: A list of tuples representing the target faces.
    Each tuple contains the uncropped facial landmarks, the cropped face image,
    the facial embedding and the cropped eyes coordinates for a single target face.
    @param eye_candidates: A list of tuples representing the source faces.
    Each tuple contains the uncropped facial landmarks, the cropped face image,
    the facial embedding and the cropped eyes coordinates for a single source face.
    :return: A tuple containing two lists. The first list contains the cropped source face images that best match each
    target face, and the second list contains the uncropped facial landmarks for each of those source faces.
    """
    source_embeddings = np.array([np.squeeze(source_embedding) for _, _, source_embedding, _ in eye_candidates])
    selected_source_faces = []
    source_landmarks = []
    for _, _, target_embedding, _ in target_faces:
        res = face_recognition.api.face_distance(source_embeddings, np.squeeze(np.array(target_embedding)))
        selected_source_faces.append(eye_candidates[np.argmin(res)][1])
        source_landmarks.append(eye_candidates[np.argmin(res)][0])
    return selected_source_faces, source_landmarks
