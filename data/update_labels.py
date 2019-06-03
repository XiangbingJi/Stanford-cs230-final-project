import os

from PIL import Image

from resizeimage import resizeimage

# directory = "/Users/loujunjie/Desktop/laneseg_label_w16/driver_161_90frame/06030819_0755.MP4/"
# ROOT_DIR = "/Users/loujunjie/Desktop/laneseg_label_w16/driver_161_90frame/06030819_0755.MP4/"
ROOT_DIR = "/Users/loujunjie/Downloads/driver_161_90frame/06030819_0755.MP4/"

def update_labels(image_parent_directory, image_filename):
    image_full_directory = os.path.join(image_parent_directory, image_filename)
#     print image_full_directory
    image = Image.open(image_full_directory)
    pixels = image.load()
    print image.size
    
    # Update labels (1,2,3,4) to 1
    for x in range(0,1640):
        for y in range(0,590):
#             print pix[x,y]
            if (pixels[x,y] == 0):
                continue;
            else:
                pixels[x,y] = 1

#     new_directory = os.path.join(image_parent_directory, "updated/")
#     new_image_name = os.path.join(new_directory, image_filename)
#     print new_image_name
    image.save(image_full_directory, image.format)
    
### Update labels
def loop_directory(directory):
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            loop_directory(os.path.join(directory, filename))
        elif filename.endswith(".png"):
#             print directory, filename
            update_labels(directory, filename)
    
if __name__ == '__main__':
    loop_directory(ROOT_DIR)