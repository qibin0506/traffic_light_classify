from sklearn.utils import shuffle
import numpy as np
import os
import cv2
import utils


def read_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (utils.IMAGE_WIDTH, utils.IMAGE_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalizer_image = image / 255.0 - 0.5

    return normalizer_image


def read_traffic_light(training=True):
    traffic_light_dir = "traffic_light_images/"

    if training:
        red = traffic_light_dir + "training/red/"
        yellow = traffic_light_dir + "training/yellow/"
        green = traffic_light_dir + "training/green/"
    else:
        red = traffic_light_dir + "test/red/"
        yellow = traffic_light_dir + "test/yellow/"
        green = traffic_light_dir + "test/green/"

    images = []
    labels = []
    image_name = []

    for f in os.listdir(red):
        images.append(read_image(red + f))
        labels.append(utils.RED)
        image_name.append(f)

    for f in os.listdir(yellow):
        images.append(read_image(yellow + f))
        labels.append(utils.YELLOW)
        image_name.append(f)

    for f in os.listdir(green):
        images.append(read_image(green + f))
        labels.append(utils.GREEN)
        image_name.append(f)

    return shuffle(np.array(images), np.array(labels), np.array(image_name))
