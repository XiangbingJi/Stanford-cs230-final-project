import os

from PIL import Image

from resizeimage import resizeimage

# directory = "/Users/loujunjie/Desktop/laneseg_label_w16/driver_161_90frame/06030819_0755.MP4/"
# ROOT_DIR = "/Users/loujunjie/Downloads/driver_161_90frame/06030819_0755.MP4/"
# ROOT_DIR = "/Users/loujunjie/Downloads/driver_161_90frame/06030819_0755.MP4/"
ROOT_DIR = "/home/ubuntu/data/copy-255/processed_tuSimple_dataset/"

# Resize to 1/5 of original size, keep same height/width ratio
resize_width = 320
resize_height = 192

def resize_image(image_parent_directory, image_filename):
#     print image_parent_directory, image_filename
    image_full_directory = os.path.join(image_parent_directory, image_filename)
#     print image_full_directory
    image = Image.open(image_full_directory)
    # print image.size

    pixels = image.load() # create the pixel map

    for i in range(resize_width): # for every pixel:
        for j in range(resize_height):
            if pixels[i,j] != (255, 255, 255, 255):
                pixels[i,j] = (0, 0 ,0, 255)
            #else:
                #pixels[i,j] = (1, 1, 1, 255)

    image.save(image_full_directory)
    print("done updating image " + image_filename)

def loop_directory(directory):
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            loop_directory(os.path.join(directory, filename))
        elif filename.endswith(".png"):
#             print directory, filename
            resize_image(directory, filename)

if __name__ == '__main__':
    loop_directory(ROOT_DIR)
