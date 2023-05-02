import os, cv2, matplotlib.pyplot as plt


def load_images(source_image_dir, target_image_path, debug=False):
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
