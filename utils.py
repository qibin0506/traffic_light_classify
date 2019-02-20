RED = 0
YELLOW = 1
GREEN = 2

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNEL = 3
IMAGE_CLASSIFY = 3

lr = 0.0001
epochs = 50
batch_size = 128

mu = 0
sigma = 0.1


def get_traffic_name(label):
    if label == RED:
        return "red"

    if label == YELLOW:
        return "yellow"

    if label == GREEN:
        return "green"
