from src import selector, replacer
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="unblink: replace closed eyes in images")
parser.add_argument("--replacement-path", "-r", help="Folder with images to use as eye sources", required=True)
parser.add_argument("--target-path", "-t", help="Image in which to replace closed eyes", required=True)
parser.add_argument("--debug", "-d", help="Enable debug output", action=argparse.BooleanOptionalAction)

def main():
    args = parser.parse_args()
    target_image_path = Path(args.target_path)
    source_image_dir = Path(args.replacement_path)
    debug = args.debug

    source_images, target_image = selector.load_images(source_image_dir, target_image_path, debug)
    target_face_landmarks, target_eyes = selector.get_eyes_from_image(target_image)
    source_eyes = [selector.get_eyes_from_image(image)[1] for image in source_images]
    source_index_to_replace_with = selector.compute_replacements(source_eyes)
    blend = replacer.replace(source_images[source_index_to_replace_with], target_image, target_face_landmarks)
    plt.imshow(blend)
    plt.show()


if __name__ == '__main__':
    main()
