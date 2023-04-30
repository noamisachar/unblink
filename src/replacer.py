import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

FACIAL_LANDMARK_PREDICTOR_PATH = 'models/shape_predictor.dat'
MINIMUM_EAR = 0.2
FACE_DETECTOR = dlib.get_frontal_face_detector()
LANDMARK_FINDER = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR_PATH)
LEFT_EYE_START, LEFT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
RIGHT_EYE_START, RIGHT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def replace(source_image, target_image, target_face_landmarks):
    """
    Replaces the eyes in the target image with the eyes from the source image.
    The source image should contain one face only, with open eyes.

    @param source_image: The source image containing the replacement eyes.
    @param target_image: The target image to replace the eyes in.
    @param target_face_landmarks: The facial landmarks of the face to replace in the target image.
    :return: The blended image with the replaced eyes.
    """
    replacement_image = match_face_size(source_image=source_image, target_face_landmarks=target_face_landmarks)
    replacement_face = FACE_DETECTOR(replacement_image, 1)[0]
    replacement_face_landmarks = face_utils.shape_to_np(LANDMARK_FINDER(replacement_image, replacement_face))
    replacement_eyes = {
        'left': replacement_face_landmarks[LEFT_EYE_START:LEFT_EYE_END],
        'right': replacement_face_landmarks[RIGHT_EYE_START:RIGHT_EYE_END]
    }
    target_eyes = {
        'left': target_face_landmarks[LEFT_EYE_START:LEFT_EYE_END],
        'right': target_face_landmarks[RIGHT_EYE_START:RIGHT_EYE_END]
    }
    replacement_surrounding_coordinates = {
        'left': [replacement_face_landmarks[23], replacement_face_landmarks[24], replacement_face_landmarks[27]],
        'right': [replacement_face_landmarks[19], replacement_face_landmarks[20], replacement_face_landmarks[27]]
    }
    target_surrounding_coordinates = {
        'left': [target_face_landmarks[23], target_face_landmarks[24], target_face_landmarks[27]],
        'right': [target_face_landmarks[19], target_face_landmarks[20], target_face_landmarks[27]]
    }
    expanded_replacement_eyes = expand_eye_coordinates(eyes_coordinates=replacement_eyes,
                                                       surrounding_coordinates=replacement_surrounding_coordinates)

    if eye_aspect_ratio(replacement_eyes['left']) < MINIMUM_EAR or \
            eye_aspect_ratio(replacement_eyes['right']) < MINIMUM_EAR:
        raise ValueError('Source images need to contain open eyes.')

    replacement_centroids = {
        'left': centroid(replacement_eyes['left']),
        'right': centroid(replacement_eyes['right'])
    }
    target_centroids = {
        'left': (
            np.int32(centroid(target_eyes['left'])[0] * 0.85 + target_surrounding_coordinates['left'][1][0] * 0.15),
            np.int32(centroid(target_eyes['left'])[1] * 0.85 + target_surrounding_coordinates['left'][1][1] * 0.15)
        ),
        'right': (
            np.int32(centroid(target_eyes['right'])[0] * 0.85 + target_surrounding_coordinates['right'][0][0] * 0.15),
            np.int32(centroid(target_eyes['right'])[1] * 0.85 + target_surrounding_coordinates['right'][0][1] * 0.15)
        )
    }
    left_eye_mask = create_eye_mask(
        replacement_image, expanded_replacement_eyes['left'], replacement_centroids['left'])
    right_eye_mask = create_eye_mask(
        replacement_image, expanded_replacement_eyes['right'], replacement_centroids['right'])

    blended_image = cv2.seamlessClone(
        replacement_image, target_image, left_eye_mask, target_centroids['left'], cv2.NORMAL_CLONE)
    blended_image = cv2.seamlessClone(
        replacement_image, blended_image, right_eye_mask, target_centroids['right'], cv2.NORMAL_CLONE)
    return blended_image


def match_face_size(source_image, target_face_landmarks):
    """
    This function matches the size of the face in the source image to the size of the face in the target image
    and returns the resized source image.

    @param source_image: The source image containing the replacement eyes.
    @param target_face_landmarks: The facial landmarks of the face to replace in the target image.
    :return: The source image resized so the face matches the size of the face in the target image.
    """
    source_faces = FACE_DETECTOR(source_image, 1)
    if len(source_faces) != 1:
        raise ValueError(f'Source images need to have a single face in them. Faces detected: {len(source_faces)}')
    source_face_landmarks = face_utils.shape_to_np(LANDMARK_FINDER(source_image, source_faces[0]))
    source_face_size = dist.euclidean(source_face_landmarks[0], source_face_landmarks[16])
    target_face_size = dist.euclidean(target_face_landmarks[0], target_face_landmarks[16])
    scaling_factor = target_face_size / source_face_size
    interpolation_type = cv2.INTER_AREA if scaling_factor < 1 else cv2.INTER_CUBIC
    replacement_image = cv2.resize(source_image, None, fx=scaling_factor, fy=scaling_factor,
                                   interpolation=interpolation_type)
    return replacement_image


def eye_aspect_ratio(eye):
    """
    This function calculates the eye aspect ratio (EAR) of a given eye.

    @param eye: The coordinates of the eye.
    :return: The eye aspect ratio.
    """
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


def centroid(eye):
    """
    This function calculates the centroid of a given eye.

    @param eye: The coordinates of the eye.
    :return: The eye's centroid coordinates.
    """
    return int(sum(eye[:, 0] / len(eye[:, ]))), int(sum(eye[:, 1] / len(eye[:, ])))


def expand_eye_coordinates(eyes_coordinates, surrounding_coordinates):
    """
    This function expands the coordinates of the eyes to by using the surrounding face coordinates.

    @param eyes_coordinates: The coordinates of the left and right eyes.
    @param surrounding_coordinates: The coordinates of the surrounding points.The coordinates of the surrounding points.
    :return: The expanded coordinates of the left and right eyes.
    """
    left_eye_coordinates = eyes_coordinates['left']
    right_eye_coordinates = eyes_coordinates['right']
    left_surrounding_coordinates = surrounding_coordinates['left']
    right_surrounding_coordinates = surrounding_coordinates['right']

    diff_left_1 = np.int32((left_surrounding_coordinates[0] * 0.35 + left_eye_coordinates[1] * 0.65)
                           - left_eye_coordinates[1])
    diff_left_2 = np.int32((left_surrounding_coordinates[1] * 0.35 + left_eye_coordinates[2] * 0.65)
                           - left_eye_coordinates[2])
    diff_left_3 = np.int32((left_surrounding_coordinates[2] * 0.35 + left_eye_coordinates[0] * 0.65)
                           - left_eye_coordinates[0])
    diff_right_1 = np.int32((right_surrounding_coordinates[0] * 0.35 + right_eye_coordinates[0] * 0.65)
                            - right_eye_coordinates[0])
    diff_right_2 = np.int32((right_surrounding_coordinates[1] * 0.35 + right_eye_coordinates[1] * 0.65)
                            - right_eye_coordinates[1])
    diff_right_3 = np.int32((right_surrounding_coordinates[2] * 0.35 + right_eye_coordinates[2] * 0.65)
                            - right_eye_coordinates[2])

    expanded_left_eye_coordinates = [left_eye_coordinates[0] + diff_left_3, left_eye_coordinates[1] + diff_left_1,
                                     left_eye_coordinates[2] + diff_left_2, left_eye_coordinates[3] - diff_left_3,
                                     left_eye_coordinates[4] - diff_left_2, left_eye_coordinates[5] - diff_left_1]
    expanded_right_eye_coordinates = [right_eye_coordinates[0] - diff_right_3, right_eye_coordinates[1] + diff_right_1,
                                      right_eye_coordinates[2] + diff_right_2, right_eye_coordinates[3] + diff_right_3,
                                      right_eye_coordinates[4] - diff_right_2, right_eye_coordinates[5] - diff_right_1]
    return {
        'left': expanded_left_eye_coordinates,
        'right': expanded_right_eye_coordinates
    }


def create_eye_mask(image, eye_coordinates, eye_centroid):
    """
    This function creates a mask which includes only the eye received.

    @param image: The image that includes the eye.
    @param eye_coordinates: The coordinates of the eye.
    @param eye_centroid: The centroid of the eye.
    :return: The mask of the eye.
    """
    eye_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    major_axis = (abs(eye_coordinates[0][0] - eye_coordinates[3][0])) / 2
    minor_axis = (abs(eye_coordinates[2][1] - eye_coordinates[4][1])) / 2
    cv2.ellipse(eye_mask, (int(eye_centroid[0]), int(eye_centroid[1])),
                (int(major_axis), int(minor_axis)), 0, 0, 360, 255, -1)
    return eye_mask
