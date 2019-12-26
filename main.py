import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from struct import *

if __name__ == '__main__':

    # Read train images and labels in binary mode.
    train_images_file = open("data/train-images.idx3-ubyte", "rb")
    train_labels_file = open("data/train-labels.idx1-ubyte", "rb")

    # Skip the header bytes of each file.
    train_images_file.read(16)
    train_labels_file.read(8)

    images = []
    labels = []
    img = None
    # Read one image and label at a time.
    while True:
        image_bytes = train_images_file.read(784)
        label_bytes = train_labels_file.read(1)

        # Break if end of file has been reached.
        if not image_bytes or not label_bytes:
            break

        # Attach extracted image and label.
        img = np.reshape(unpack(len(image_bytes) * "B", image_bytes), (28, 28))
        images.append(img)
        lbl = int(label_bytes[0])
        labels.append(lbl)

    # Display last images for debugging.
    plt.imshow(img, cmap="gray")
    plt.show()
