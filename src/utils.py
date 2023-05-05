import cv2
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance as dist
import numpy as np
import dlib
from imutils import face_utils


MINIMUM_EAR = 0.21
FACE_DETECTOR = dlib.get_frontal_face_detector()
LEFT_EYE_START, LEFT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
RIGHT_EYE_START, RIGHT_EYE_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def load_images(source_image_dir, target_image_path, debug=False):
    """
    Loads the target and source images from the given paths.

    @param source_image_dir: The path of the source images.
    @param target_image_path: The path of the target image.
    @param debug: If true - prints the images.
    :return: A list of source images and a target image.
    """
    if not target_image_path.exists():
        return ValueError("The target image doesn't exist")
    if not source_image_dir.exists():
        return ValueError("The source image directory doesn't exist")

    source_images = []
    for image in source_image_dir.iterdir():
        image = cv2.cvtColor(
            cv2.imread(os.path.join(source_image_dir, image.name)), cv2.COLOR_BGR2RGB
        )
        if debug:
            plt.imshow(image)
            plt.title("Loaded source image")
            plt.show()
        source_images.append(image)

    if len(source_images) == 0:
        return ValueError("Source image directory is empty")

    target_image = cv2.cvtColor(cv2.imread(str(target_image_path)), cv2.COLOR_BGR2RGB)
    if debug:
        plt.imshow(target_image)
        plt.title("Loaded target image")
        plt.show()

    return source_images, target_image


def centroid(eye):
    """
    This function calculates the centroid of a given eye.

    @param eye: The coordinates of the eye.
    :return: The eye's centroid coordinates.
    """
    return int(sum(eye[:, 0] / len(eye[:, ]))), int(sum(eye[:, 1] / len(eye[:, ])))


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


def get_eyes_and_surrounding_coordinates(face_landmarks):
    """
    This function returns the coordinates that are relevant to us, the eyes and the surroundings.

    @param face_landmarks: Face landmarks coordinates.
    :return: Dictionaries holding the eyes and surrounding coordinates.
    """
    eyes_coordinates = {
        'left': face_landmarks[LEFT_EYE_START:LEFT_EYE_END],
        'right': face_landmarks[RIGHT_EYE_START:RIGHT_EYE_END]
    }
    surrounding_coordinates = {
        'left': [face_landmarks[23], face_landmarks[24], face_landmarks[27]],
        'right': [face_landmarks[19], face_landmarks[20], face_landmarks[27]]
    }
    return eyes_coordinates, surrounding_coordinates


def expand_eye_coordinates(eyes_coordinates, surrounding_coordinates):
    """
    This function expands the coordinates of the eyes by using the surrounding face coordinates.

    @param eyes_coordinates: The coordinates of the left and right eyes.
    @param surrounding_coordinates: The coordinates of the surrounding points.The coordinates of the surrounding points.
    :return: The expanded coordinates of the left and right eyes.
    """
    left_eye_coordinates = eyes_coordinates['left']
    right_eye_coordinates = eyes_coordinates['right']
    left_surrounding_coordinates = surrounding_coordinates['left']
    right_surrounding_coordinates = surrounding_coordinates['right']

    diff_left_1 = np.int32((left_surrounding_coordinates[0] * 0.4 + left_eye_coordinates[1] * 0.6)
                           - left_eye_coordinates[1])
    diff_left_2 = np.int32((left_surrounding_coordinates[1] * 0.4 + left_eye_coordinates[2] * 0.6)
                           - left_eye_coordinates[2])
    diff_left_3 = np.int32((left_surrounding_coordinates[2] * 0.4 + left_eye_coordinates[0] * 0.6)
                           - left_eye_coordinates[0])
    diff_right_1 = np.int32((right_surrounding_coordinates[0] * 0.4 + right_eye_coordinates[0] * 0.6)
                            - right_eye_coordinates[0])
    diff_right_2 = np.int32((right_surrounding_coordinates[1] * 0.4 + right_eye_coordinates[1] * 0.6)
                            - right_eye_coordinates[1])
    diff_right_3 = np.int32((right_surrounding_coordinates[2] * 0.4 + right_eye_coordinates[2] * 0.6)
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
