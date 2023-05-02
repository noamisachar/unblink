# unblink

## Description
Automatically detects and replaces closed or blinked eyes in images.

## Installation
* Clone this repository to your local machine.
* Install the required dependencies by running the following command in your terminal: `pip install -r requirements.txt`

## Usage
Run the following command from the project directory to start the program:

`python main.py --target-path path/to/target/image.jpg --replacement-path path/to/source/images`

Replace `path/to/target/image.jpg` with the path to the image where you want to replace the eyes, and `path/to/source/images` with the path to the folder containing the replacement eyes images.
If you don't have any replacement images, the program will use the default images that we provided.

The program will output a new image with the replaced eyes in the same directory as the target image.

## Example
ADD ACTUAL EXAMPLE
`python main.py -i path/to/image.jpg -s path/to/source/images`
This will replace the closed eyes in image.jpg with the most similar open eyes from the images in the source/images folder, and save the output image in the same directory as image.jpg.

## Acknowledgements
This program was created by Noam Isachar, Ilya Andreev and Raj Krishnan as part of a project for CS 445 - Computational Photography course at University of Illinois Urbana-Champaign.
