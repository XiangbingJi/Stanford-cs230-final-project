from PIL import Image
import glob

original_images = []

ori_glob = glob.glob('frames/*.jpg')
mark_glob = glob.glob('predicted_frames/*.jpg')

for i in range(len(ori_glob)):
    background = Image.open(ori_glob[i])
    overlay = Image.open(mark_glob[i])

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.5)
    new_img.save("overlay/" + str(i) + ".png", "PNG")

