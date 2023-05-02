import cv2
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import utils


def replace(source_images, target_image, target_face_landmarks, facial_landmark_predictor):
    """
    Replaces all closed eyes in the target image with the eyes from the source image.
    The source image should contain one face only, with open eyes.

    @param source_images: The source images containing the replacement eyes.
    @param target_image: The target image to replace the eyes in.
    @param target_face_landmarks: The facial landmarks of the face to replace in the target image.
    @param facial_landmark_predictor: The model predicting locations of faces and facial features.
    :return: The blended image with the replaced eyes.
    """
    print(f"Attempting to replace eyes in {len(target_face_landmarks)} faces")
    for [_, landmark, _, eyes], source_image in zip(target_face_landmarks, source_images):
        print(eyes)
        # If the target eyes aren't closed, then the function call is a no-op.
        if eyes[0]['EAR'] > utils.MINIMUM_EAR or eyes[1]['EAR'] > utils.MINIMUM_EAR:
            print("skipping target")
            continue
        target_image = replace_inner(source_image, target_image, landmark, facial_landmark_predictor)
    return target_image


def replace_inner(source_image, target_image, target_face_landmarks, facial_landmark_predictor):
    """
    Replaces the eyes in the target image with the eyes from the source image.
    The source image should contain one face only, with open eyes.

    @param source_image: The source image containing the replacement eyes.
    @param target_image: The target image to replace the eyes in.
    @param target_face_landmarks: The facial landmarks of the face to replace in the target image.
    @param facial_landmark_predictor: The model predicting locations of faces and facial features.
    :return: The blended image with the replaced eyes.
    """

    replacement_image = match_face_size(source_image=source_image, target_face_landmarks=target_face_landmarks,
                                        facial_landmark_predictor=facial_landmark_predictor)
    replacement_face = utils.FACE_DETECTOR(replacement_image, 1)[0]
    replacement_face_landmarks = face_utils.shape_to_np(facial_landmark_predictor(replacement_image, replacement_face))
    replacement_eyes, replacement_surrounding_coordinates = utils.get_eyes_and_surrounding_coordinates(
        replacement_face_landmarks)
    target_eyes, target_surrounding_coordinates = utils.get_eyes_and_surrounding_coordinates(
        target_face_landmarks)
    expanded_replacement_eyes = utils.expand_eye_coordinates(
        eyes_coordinates=replacement_eyes, surrounding_coordinates=replacement_surrounding_coordinates)

    if utils.eye_aspect_ratio(replacement_eyes['left']) < utils.MINIMUM_EAR or \
            utils.eye_aspect_ratio(replacement_eyes['right']) < utils.MINIMUM_EAR:
        raise ValueError('Source image needs to contain open eyes.')

    replacement_centroids = {
        'left': utils.centroid(replacement_eyes['left']),
        'right': utils.centroid(replacement_eyes['right'])
    }
    target_centroids = {
        'left': (
            np.int32(utils.centroid(target_eyes['left'])[0] * 0.85
                     + target_surrounding_coordinates['left'][1][0] * 0.15),
            np.int32(utils.centroid(target_eyes['left'])[1] * 0.85
                     + target_surrounding_coordinates['left'][1][1] * 0.15)
        ),
        'right': (
            np.int32(utils.centroid(target_eyes['right'])[0] * 0.85
                     + target_surrounding_coordinates['right'][0][0] * 0.15),
            np.int32(utils.centroid(target_eyes['right'])[1] * 0.85
                     + target_surrounding_coordinates['right'][0][1] * 0.15)
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


def match_face_size(source_image, target_face_landmarks, facial_landmark_predictor):
    """
    This function matches the size of the face in the source image to the size of the face in the target image
    and returns the resized source image.

    @param source_image: The source image containing the replacement eyes.
    @param target_face_landmarks: The facial landmarks of the face to replace in the target image.
    @param facial_landmark_predictor: The model predicting locations of faces and facial features.
    :return: The source image resized so the face matches the size of the face in the target image.
    """
    source_faces = utils.FACE_DETECTOR(source_image, 1)
    if len(source_faces) != 1:
        raise ValueError(f'Source images need to have a single face in them. Faces detected: {len(source_faces)}')
    source_face_landmarks = face_utils.shape_to_np(facial_landmark_predictor(source_image, source_faces[0]))
    source_alignment_points = [source_face_landmarks[0], source_face_landmarks[16]]
    target_alignment_points = [target_face_landmarks[0], target_face_landmarks[16]]

    source_face_size = dist.euclidean(source_alignment_points[0], source_alignment_points[1])
    target_face_size = dist.euclidean(target_alignment_points[0], target_alignment_points[1])
    scaling_factor = target_face_size / source_face_size
    interpolation_type = cv2.INTER_AREA if scaling_factor < 1 else cv2.INTER_CUBIC
    replacement_image = cv2.resize(source_image, None, fx=scaling_factor, fy=scaling_factor,
                                   interpolation=interpolation_type)

    source_dy = source_alignment_points[1][1] - source_alignment_points[0][1]
    source_dx = source_alignment_points[1][0] - source_alignment_points[0][0]
    source_angle = np.degrees(np.arctan2(source_dy, source_dx))
    target_dy = target_alignment_points[1][1] - target_alignment_points[0][1]
    target_dx = target_alignment_points[1][0] - target_alignment_points[0][0]
    target_angle = np.degrees(np.arctan2(target_dy, target_dx))
    replacement_image = imutils.rotate(replacement_image, source_angle - target_angle)

    return replacement_image


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
