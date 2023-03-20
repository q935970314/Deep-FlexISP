import os

import imageio

PATH_TO_IMAGES = os.path.join("vis", "gif_frames_2k_epochs", "IMG_0753")
GIF_NAME = "test_400_epochs"


def main():
    file_names = [file_name for file_name in os.listdir(PATH_TO_IMAGES)]
    print(file_names)
    file_names.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    print("\nGenerating GIF from {} images stored at {}\n".format(len(file_names), PATH_TO_IMAGES))

    images = []
    for file_name in file_names:
        print("\t Frame: {}".format(file_name))
        images += [imageio.imread(os.path.join(PATH_TO_IMAGES, file_name))]

    print("\n Generating GIF at {}... \n".format(PATH_TO_IMAGES))
    imageio.mimsave(os.path.join(PATH_TO_IMAGES, "{}.gif".format(GIF_NAME)), images)
    print(" GIF generated successfully! \n")


if __name__ == '__main__':
    main()
