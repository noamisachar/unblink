import dlib, cv2, matplotlib.pyplot as plt, numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import face_recognition
import math

MINIMUM_EAR = 0.21
face_detector = dlib.get_frontal_face_detector()
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Accept an image and locate the eyes within it. Compute eye coordinates and
# classify each eye as open or closed. Return two objects for every detected
# face: the left eye and the right eye.
def get_all_faces_and_eyes_from_image(image, facial_landmark_predictor, debug):
    faces = face_detector(image, 1)
    print("faces in get_all_faces_and_eyes_from_image ", len(faces))

    # No cropping required
    if len(faces) == 1:
        face, face_landmarks, eyes = _get_eyes_from_image(image, facial_landmark_predictor)
        if len(eyes) != 2:
            print("Skipping bc too few eyes ", face_landmarks, eyes)
            return []
        face_embedding = face_recognition.face_encodings(image)
        return [(image, face_landmarks, face_embedding, eyes)]
 
    # Crop to each face separately.
    results = []
    for face in faces:
        cropped_image = _get_cropped_face(image, face, padding_pct=100, debug=debug)
        face, face_landmarks, eyes = _get_eyes_from_image(cropped_image, facial_landmark_predictor)
        if len(eyes) != 2:
            print("Skipping bc too few eyes ", face_landmarks, eyes)
            continue
        face_embedding = face_recognition.face_encodings(cropped_image)
        results.append((cropped_image, face_landmarks, face_embedding, eyes))

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

        faces = face_detector(cropped_image, 1)
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
    faces = face_detector(image, 1)
    face = None
    if len(faces) >= 1:
        face = faces[0]
    elif len(faces) == 0:
        print("No faces found in image")
        return (None, None, ())

    face_landmarks = facial_landmark_predictor(image, face)
    face_landmarks = face_utils.shape_to_np(face_landmarks)
    left_eye = face_landmarks[left_eye_start:left_eye_end]
    right_eye = face_landmarks[right_eye_start:right_eye_end]
    left_EAR = _eye_aspect_ratio(left_eye)
    right_EAR = _eye_aspect_ratio(right_eye)
 
    # need to make these more robust and also for centroids
    diff_left_1 = np.int32((face_landmarks[23] + face_landmarks[43]) / 2) - face_landmarks[43]
    diff_left_2 = np.int32((face_landmarks[24] + face_landmarks[44]) / 2) - face_landmarks[44]
    diff_left_3 = np.int32((face_landmarks[27] + face_landmarks[42]) / 2) - face_landmarks[42]
    diff_right_1 = np.int32((face_landmarks[19] + face_landmarks[37]) / 2) - face_landmarks[37]
    diff_right_2 = np.int32((face_landmarks[20] + face_landmarks[38]) / 2) - face_landmarks[38]
    diff_right_3 = np.int32((face_landmarks[27] + face_landmarks[39]) / 2) - face_landmarks[39]

    left_eye[0] += diff_left_3
    left_eye[1] += diff_left_1
    left_eye[2] += diff_left_2
    left_eye[3] -= diff_left_3
    left_eye[4] -= diff_left_2
    left_eye[5] -= diff_left_1

    right_eye[0] -= diff_right_3
    right_eye[1] += diff_right_1
    right_eye[2] += diff_right_2
    right_eye[3] += diff_right_3
    right_eye[4] -= diff_right_2
    right_eye[5] -= diff_right_1

    left_eye = {
        "kind": "left",
        "EAR": left_EAR,
        "status": "closed" if left_EAR < MINIMUM_EAR else "open",
        "centroid": _centroid(left_eye),
        "coordinates": left_eye,
    }

    right_eye = {
        "kind": "right",
        "EAR": right_EAR,
        "status": "closed" if right_EAR < MINIMUM_EAR else "open",
        "centroid": _centroid(right_eye),
        "coordinates": right_eye,
    }

    return face, face_landmarks, (left_eye, right_eye)


# Compute EAR, a good indicator of whether an eye is open or closed. Source:
# Source: https://www.mdpi.com/2079-9292/11/19/3183.
# 
# EAR = (|| P2 - P6 || + || P3 - P5 ||) / 2 * || P1 - P4 ||
def _eye_aspect_ratio(eye):
    return (dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])) / (2.0 * dist.euclidean(eye[0], eye[3]))


# Centroids are used to compute the target blending position.
def _centroid(eye):
    return int(sum(eye[:, 0] / len(eye[:, ]))), int(sum(eye[:, 1] / len(eye[:, ])))


# Identify the source face which is most similar to the provided target faces
def compute_replacements(target_faces, eye_candidates):
    source_embeddings = np.array([np.squeeze(source_embedding) for _, _, source_embedding, _ in eye_candidates])
    selected_source_faces = []
    for _, _, target_embedding, _ in target_faces:
        res = face_recognition.api.face_distance(source_embeddings, np.squeeze(np.array(target_embedding)))
        selected_source_faces.append(eye_candidates[np.argmin(res)][0])
    return selected_source_faces
