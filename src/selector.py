import math
import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from . import utils


# Accept an image and locate the eyes within it. Compute eye coordinates and
# classify each eye as open or closed. Return two objects for every detected
# face: the left eye and the right eye.
def get_all_faces_and_eyes_from_image(image, facial_landmark_predictor, debug):
    faces = utils.FACE_DETECTOR(image, 1)
    print("faces in get_all_faces_and_eyes_from_image ", len(faces))
 
    # Crop to each face separately.
    results = []
    for face in faces:
        cropped_image = _get_cropped_face(image, face, padding_pct=15, debug=debug)
        cropped_face, cropped_face_landmarks, cropped_eyes = _get_eyes_from_image(
            cropped_image, facial_landmark_predictor)
        if len(cropped_eyes) != 2:
            print("Skipping bc too few eyes ", cropped_face_landmarks, cropped_eyes)
            continue
        face_embedding = face_recognition.face_encodings(cropped_image)[0]
        uncropped_landmarks = face_utils.shape_to_np(facial_landmark_predictor(image, face))
        results.append((uncropped_landmarks, cropped_image, cropped_face_landmarks, face_embedding, cropped_eyes))

    return results


# Crop the image to include just the provided rectangle, along with some padding
def _get_cropped_face(image, face_rect, padding_pct=50, debug=False):
    x, y, w, h = face_utils.rect_to_bb(face_rect)
    padding = math.floor((padding_pct / 100) * max(w, h))

    x_min = max(0, x - padding)
    x_max = min(x + w + padding, image.shape[1])
    y_min = max(0, y - padding)
    y_max = min(y + h + padding, image.shape[0])

    # Copy the array to a new image, as numpy <> dlib's interaction does not work
    # well with slices
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


# Accept an image and locate the eyes within it. Compute eye coordinates and
# classify each eye as open or closed. Return two objects: the left eye and the
# right eye.
def _get_eyes_from_image(image, facial_landmark_predictor):
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


# Identify the source face which is most similar to the provided target faces
def compute_replacements(target_faces, eye_candidates):
    source_embeddings = np.array([np.squeeze(source_embedding) for _, _, _, source_embedding, _ in eye_candidates])
    selected_source_faces = []
    source_landmarks = []
    for _, _, _, target_embedding, _ in target_faces:
        res = face_recognition.api.face_distance(source_embeddings, np.squeeze(np.array(target_embedding)))
        selected_source_faces.append(eye_candidates[np.argmin(res)][1])
        source_landmarks.append(eye_candidates[np.argmin(res)][0])
    return selected_source_faces, source_landmarks
