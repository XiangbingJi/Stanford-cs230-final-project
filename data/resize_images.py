import os

from PIL import Image

from resizeimage import resizeimage

# directory = "/Users/loujunjie/Desktop/laneseg_label_w16/driver_161_90frame/06030819_0755.MP4/"
# ROOT_DIR = "/Users/loujunjie/Downloads/driver_161_90frame/06030819_0755.MP4/"
# ROOT_DIR = "/Users/loujunjie/Downloads/driver_161_90frame/06030819_0755.MP4/"
ROOT_DIR = "/home/ubuntu/git_repos/cs230/Stanford-cs230-final-project/utils/video_frames"

# Resize to 1/5 of original size, keep same height/width ratio
resize_width = 320
resize_height = 192

def resize_image(image_parent_directory, image_filename):
#     print image_parent_directory, image_filename
    image_full_directory = os.path.join(image_parent_directory, image_filename)
#     print image_full_directory
    image = Image.open(image_full_directory)
    # print image.size
    cover = resizeimage.resize_cover(image, [resize_width, resize_height], validate = False)
    
    pixels = image.load() # create the pixel map

    cover.save(image_full_directory, image.format)
    print("done resizing image " + image_filename)
    
def loop_directory(directory):
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            loop_directory(os.path.join(directory, filename))
        elif filename.endswith(".jpg"):
#             print directory, filename
            resize_image(directory, filename)
    
if __name__ == '__main__':
    loop_directory(ROOT_DIR)
