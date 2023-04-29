import argparse, dlib, os, cv2, matplotlib.pyplot as plt, numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from . import replacer

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
        print("The target image doesn't exist")
        raise SystemExit(1)
    if not source_image_dir.exists():
        print("The source image directory doesn't exist")
        raise SystemExit(1)
    
    source_images = []
    for image in source_image_dir.iterdir():
        image = cv2.cvtColor(cv2.imread(os.path.join(source_image_dir, image.name)), cv2.COLOR_BGR2GRAY)
        if debug:
            plt.imshow(image, cmap="gray")
            plt.show()
        source_images.append(image)
        
    if len(source_images) == 0:
        print("Source image directory is empty")
        raise SystemExit(1)
    
    target_image = cv2.cvtColor(cv2.imread(str(target_image_path)), cv2.COLOR_BGR2GRAY)
    if debug:
        plt.imshow(target_image, cmap="gray")
        plt.show()

    return source_images, target_image


# Accept an image and locate the eyes within it. Compute eye coordinates and
# classify each eye as open or closed. Return two objects: the left eye and the
# right eye.
def get_eyes_from_image(image):
    faces = face_detector(image, 0)
    # For now only support if and only if there is one face in the image.
    assert(len(faces) == 1)
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
        "EAR": left_EAR,
        "status": "closed" if right_EAR < MINIMUM_EAR else "open",
        "centroid": centroid(right_eye),
        "coordinates": right_eye,
    }
    
    return left_eye, right_eye


def compute_replacements(target_eyes, source_eyes):
    replacements = []
    for i, target_eye in enumerate(target_eyes):
        if target_eye['status'] == 'open':
            continue
        
        max_source_EAR = -1
        source_eye = None
        source_image_index = -1
        for j, candidate_eyes in enumerate(source_eyes):
            candidate_eye = candidate_eyes[i]
            if candidate_eye['status'] == 'open' and candidate_eye['EAR'] > max_source_EAR:
                max_source_EAR = candidate_eye['EAR']
                source_eye = candidate_eye
                source_image_index = j
        
        if source_eye is None:
            # We have nothing to replace this particular eye with, continue.
            continue
        
        replacements.append({
            'target_eye': target_eye,
            'source_image_index': source_image_index,
            'source_eye': source_eye
        })

    return replacements


def main():
    args = parser.parse_args()
    target_image_path = Path(args.target_image)
    source_image_dir = Path(args.source_images_folder)
    debug = args.debug
    faceDetector = dlib.get_frontal_face_detector()
    landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
    (leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    source_images, target_image = load_images(source_image_dir, target_image_path, debug)
    
    target_eyes = get_eyes_from_image(target_image)
    source_eyes = [get_eyes_from_image(image) for image in source_images]
    
    replacements = compute_replacements(target_eyes, source_eyes)
    print(replacements)
    

if __name__ == "__main__":
    main()
    
