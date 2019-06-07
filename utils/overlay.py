from PIL import Image
import glob
import argparse
import natsort
import os.path as path


def generate_overlay(background_path, overlay_path, output_path):

    ori_glob = glob.glob(background_path + '/*.jpg') + glob.glob(background_path + '/*.png')
    mark_glob = glob.glob(overlay_path + '/*.jpg') + glob.glob(overlay_path + '/*.png')

    print(len(ori_glob), len(mark_glob))

    for i in range(len(ori_glob)):
        background = Image.open(ori_glob[i])
        overlay = Image.open(mark_glob[i])

        background = background.convert("RGBA")
        overlay = overlay.convert("RGBA")

        new_img = Image.blend(background, overlay, 0.5)
        print("saving to " + output_path + " as " + path.basename(ori_glob[i]))
        new_img.save(path.join(output_path, path.basename(ori_glob[i])), "PNG")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--background_path", type=str, default="frames")
    parser.add_argument("--overlay_path", type=str, default="predicted_frames")
    parser.add_argument("--output_path", type=str, default="overlay")

    args = parser.parse_args()

    generate_overlay(args.background_path, args.overlay_path, args.output_path)
