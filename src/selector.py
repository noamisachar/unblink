import argparse, dlib, os, cv2, matplotlib.pyplot as plt, numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

FACIAL_LANDMARK_PREDICTOR = "models/shape_predictor.dat"  
MINIMUM_EAR = 0.2
face_detector = dlib.get_frontal_face_detector()
landmark_finder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Compute EAR, a good indicator of whether an eye is open or closed. Source:
# Source: https://www.mdpi.com/2079-9292/11/19/3183.
# EAR = (|| P2 - P6 || + || P3 - P5 ||) / 2 * || P1 - P4 ||
def eye_aspect_ratio(eye):
    return (dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])) / (2.0 * dist.euclidean(eye[0], eye[3]))


# Centroids are used to compute the target blending position.
def centroid(eye):
    return int(sum(eye[:, 0] / len(eye[:, ]))), int(sum(eye[:, 1] / len(eye[:, ])))


def load_images(source_image_dir, target_image_path, debug=False):
    if not target_image_path.exists():
        return ValueError("The target image doesn't exist")
    if not source_image_dir.exists():
        return ValueError("The source image directory doesn't exist")
    
    source_images = []
    for image in source_image_dir.iterdir():
        image = cv2.cvtColor(cv2.imread(os.path.join(source_image_dir, image.name)), cv2.COLOR_BGR2GRAY)
        if debug:
            plt.imshow(image, cmap="gray")
            plt.show()
        source_images.append(image)
        
    if len(source_images) == 0:
        return ValueError("Source image directory is empty")
    
    target_image = cv2.cvtColor(cv2.imread(str(target_image_path)), cv2.COLOR_BGR2GRAY)
    if debug:
        plt.imshow(target_image, cmap="gray")
        plt.show()

    return source_images, target_image


# Accept an image and locate the eyes within it. Compute eye coordinates and
# classify each eye as open or closed. Return two objects: the left eye and the
# right eye.
def get_eyes_from_image(image):
    faces = face_detector(image, 1)

    # We expect to have exactly one face in this function
    if len(faces) != 1:
        return (None, ())

    face_landmarks = landmark_finder(image, faces[0])
    face_landmarks = face_utils.shape_to_np(face_landmarks)
    left_eye = face_landmarks[left_eye_start:left_eye_end]
    right_eye = face_landmarks[right_eye_start:right_eye_end]
    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)
    
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
        "centroid": centroid(left_eye),
        "coordinates": left_eye,
    }
    
    right_eye = {
        "kind": "right",
        "EAR": right_EAR,
        "status": "closed" if right_EAR < MINIMUM_EAR else "open",
        "centroid": centroid(right_eye),
        "coordinates": right_eye,
    }
    
    return face_landmarks, (left_eye, right_eye)

# Accept an image and locate the eyes within it. Compute eye coordinates and
# classify each eye as open or closed. Return two objects for every detected
# face: the left eye and the right eye.
def get_all_faces_and_eyes_from_image(image):
    faces = face_detector(image, 1)
    results = []

    for face in faces:
        cropped_image = get_cropped_face(image, face, padding=60)
        _, eyes = get_eyes_from_image(cropped_image)

        if len(eyes) != 2:
            continue

        results.append((cropped_image, eyes))

    return results

def get_cropped_face(image, face_rect, padding=10):
    x, y, w, h = face_utils.rect_to_bb(face_rect)

    x_min = max(0, x - padding)
    x_max = min(x + w + padding, image.shape[0])
    y_min = max(0, y - padding)
    y_max = min(y + h + padding, image.shape[1])
    
    return image[x_min:x_max, y_min:y_max]

def compute_replacements(source_eyes):
    if len(source_eyes) < 1:
        raise ValueError("source_eyes length can't be less than 1")
    
    return sorted(range(len(source_eyes)), key=lambda k: 1 / (source_eyes[k][0]['EAR'] * source_eyes[k][1]['EAR']))[0]