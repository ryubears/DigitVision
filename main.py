import numpy as np
import matplotlib.pyplot as plt
from struct import *


# Reference: https://m.blog.naver.com/PostView.nhn?blogId=acwboy&logNo=220584307823&proxyReferer=https%3A%2F%2Fwww.google.com%2F
def extract_images_and_labels():

    # Read train images and labels in binary mode.
    train_images_file = open("data/train-images.idx3-ubyte", "rb")
    train_labels_file = open("data/train-labels.idx1-ubyte", "rb")
    test_images_file = open("data/t10k-images.idx3-ubyte", "rb")
    test_labels_file = open("data/t10k-labels.idx1-ubyte", "rb")

    # Skip the header bytes of each file.
    train_images_file.read(16)
    train_labels_file.read(8)
    test_images_file.read(16)
    test_labels_file.read(8)

    train_image_list = []
    train_label_list = []
    # Read one train image and label at a time.
    while True:
        image_bytes = train_images_file.read(784)
        label_bytes = train_labels_file.read(1)

        # Break if end of file has been reached.
        if not image_bytes or not label_bytes:
            break

        # Attach extracted image and label.
        img = np.reshape(unpack(len(image_bytes) * "B", image_bytes), (28, 28))
        train_image_list.append(img)
        lbl = int(label_bytes[0])
        train_label_list.append(lbl)

    test_image_list = []
    test_label_list = []
    # Read one test image and label at a time.
    while True:
        image_bytes = test_images_file.read(784)
        label_bytes = test_labels_file.read(1)

        # Break if end of file has been reached.
        if not image_bytes or not label_bytes:
            break

        # Attach extracted image and label.
        img = np.reshape(unpack(len(image_bytes) * "B", image_bytes), (28, 28))
        test_image_list.append(img)
        lbl = int(label_bytes[0])
        test_label_list.append(lbl)

    return train_image_list, train_label_list, test_image_list, test_label_list


if __name__ == '__main__':

    # Extract images and labels from byte files.
    train_images, train_labels, test_images, test_labels = extract_images_and_labels()

    # DEBUG.
    plt.imshow(train_images[0], cmap="gray")
    plt.show()
    print(train_labels[0])
    plt.imshow(test_images[0], cmap="gray")
    plt.show()
    print(test_labels[0])
    print(len(train_images))
    print(len(test_images))
