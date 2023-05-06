# unblink

Automatically detects and replaces closed or blinked eyes in images.

![closed_1](https://user-images.githubusercontent.com/53346912/236633186-8e2e5c1e-848a-44e4-b499-64dda49fee8c.jpg)
![blend](https://user-images.githubusercontent.com/53346912/236633188-49b7df55-89f2-46e5-8caf-1388c04ed165.jpg)


## Installation
* Clone this repository to your local machine.
* Install dlib using [this](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f) guide.
* Install the required dependencies by running the following command in your terminal: `pip install -r requirements.txt`

## Usage
Run the following command from the project directory to start the program:

`python unblink.py --target-path path/to/target/image.jpg --replacement-path path/to/source/images`

Replace `path/to/target/image.jpg` with the path to the image where you want to replace the eyes, and `path/to/source/images` with the path to the folder containing the replacement eyes images.

The program will output a new image with the replaced eyes in the same directory as the target image.

## Example
`python unblink.py --target-path example/closed/closed_1.jpg --replacement-path example/open`
will replace the closed eyes in the target image with the most similar open eyes from the images in the open eyes sources folder, and save the output image in the root directory.

## Acknowledgements
This program was created by Noam Isachar, Ilya Andreev and Raj Krishnan as part of a project for CS 445 - Computational Photography course at University of Illinois Urbana-Champaign.
