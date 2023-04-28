from src import selector
import argparse


def main():
    parser = argparse.ArgumentParser(description='unblink: replace closed eyes in images')
    parser.add_argument('--target_path', '-t',
                        help='Path of the image where eye replacement is desired.',
                        required=True)
    parser.add_argument('--replacement_path', '-r',
                        help='Path that holds images with eyes to use as replacement. If empty, default image is used.',
                        required=False)
    args = parser.parse_args()

    outputs = selector(args.arg1)

    print(outputs)


if __name__ == '__main__':
    main()
