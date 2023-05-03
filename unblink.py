import cv2

from src import selector, replacer, utils
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import dlib


FACIAL_LANDMARK_PREDICTOR_PATH = "models/shape_predictor.dat"
FACIAL_LANDMARK_PREDICTOR = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR_PATH)


parser = argparse.ArgumentParser(description="unblink: replace closed eyes in images")
parser.add_argument("--target-path", "-t", help="Image in which to replace closed eyes",
                    required=True)
parser.add_argument("--replacement-path", "-r", help="Folder with images to use as eye sources",
                    default="example/open")
parser.add_argument("--debug", "-d", help="Enable debug output", action=argparse.BooleanOptionalAction)


def main():
    args = parser.parse_args()
    target_image_path = Path(args.target_path)
    source_image_dir = Path(args.replacement_path)
    debug = args.debug

    source_images, target_image = utils.load_images(
        source_image_dir, target_image_path, debug
    )

    eye_candidates = []
    for image in source_images:
        detected_faces_and_eyes = selector.get_all_faces_and_eyes_from_image(image, FACIAL_LANDMARK_PREDICTOR, debug)
        for face_and_eyes in detected_faces_and_eyes:
            if face_and_eyes[4][0]['EAR'] > utils.MINIMUM_EAR and face_and_eyes[4][1]['EAR'] > utils.MINIMUM_EAR:
                eye_candidates.append(face_and_eyes)

    target_face_landmarks = selector.get_all_faces_and_eyes_from_image(target_image, FACIAL_LANDMARK_PREDICTOR, debug)
    source_images_to_replace_with, source_landmarks = selector.compute_replacements(
        target_face_landmarks, eye_candidates)

    blend = replacer.replace(
        source_images_to_replace_with,
        source_landmarks,
        target_image,
        target_face_landmarks,
        FACIAL_LANDMARK_PREDICTOR
    )

    plt.imshow(blend)
    plt.show()

    cv2.imwrite('blend.jpg', cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
