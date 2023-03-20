import os 
import cv2 as cv

inp_root = "./results/HAT_GAN_Real_SRx4/visualization/custom"

for inp in os.listdir(inp_root):
    img = cv.imread(os.path.join(inp_root, inp))
    cv.imwrite(os.path.join("/data", "{}.jpg".format(inp.replace("_HAT_GAN_Real_SRx4.png", ""))), img, [cv.IMWRITE_JPEG_QUALITY, 100])
