from src import selector, replacer
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="unblink: replace closed eyes in images")
parser.add_argument("--replacement-path", "-r", help="Folder with images to use as eye sources", required=True)
parser.add_argument("--target-path", "-t", help="Image in which to replace closed eyes", required=True)
parser.add_argument("--debug", "-d", help="Enable debug output", action=argparse.BooleanOptionalAction)
parser.add_argument("--multi", "-m", help="Run the replacer in multiface mode", required=True, default=False)

def main():
    args = parser.parse_args()
    target_image_path = Path(args.target_path)
    source_image_dir = Path(args.replacement_path)
    debug = args.debug
    multi = args.multi

    source_images, target_image = selector.load_images(
        source_image_dir, target_image_path, debug
    )
    eye_candidates = []
    for image in source_images:
        detected_faces_and_eyes = selector.get_all_faces_and_eyes_from_image(image)
        eye_candidates.extend(detected_faces_and_eyes)

    blend = None
    if multi:
        target_face_landmarks = selector.get_all_faces_and_eyes_from_image(target_image)
        source_index_to_replace_with = selector.compute_replacements(
            [eyes for _, _, eyes in eye_candidates]
        )

        print(
            f"Selected Image #{source_index_to_replace_with} out of {len(eye_candidates)} images"
        )
        blend = replacer.replace(
            eye_candidates[source_index_to_replace_with][0],
            target_image,
            selector.get_faces_from_image(target_image),
        )
    else:
        target_face_landmarks, _ = selector.get_eyes_from_image(target_image)

        source_index_to_replace_with = selector.compute_replacements(
            [eyes for _, _, eyes in eye_candidates]
        )
        print(
            f"Selected Image #{source_index_to_replace_with} out of {len(eye_candidates)} images"
        )
        blend = replacer.replace_inner(
            eye_candidates[source_index_to_replace_with][0],
            target_image,
            target_face_landmarks,
        )

    plt.imshow(blend)
    plt.show()


if __name__ == '__main__':
    main()
