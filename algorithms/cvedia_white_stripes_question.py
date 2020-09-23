import numpy as np
import cv2


def create_white_stripes_image(width, height, stripe_width):
    img = np.zeros(shape=(height, width), dtype=np.uint8)
    stripe_coords = []
    curr_column_index = stripe_width
    # We create column coordinates by jumping two times stripe_width at each iteration, up to image width.
    while curr_column_index < width:
        stripe_coords.append(np.arange(curr_column_index, curr_column_index + stripe_width))
        curr_column_index += 2 * stripe_width
    # We concatenate all column indices into a single numpy array
    stripe_coords = np.concatenate(stripe_coords)
    # We eliminate coordinates on the right side of the image width.
    stripe_coords = stripe_coords[stripe_coords < width]
    # We set all column pixels, corresponding to column indices in the "stripe_coords" array to 255.
    img[:, stripe_coords] = 255
    # Write the image into a file.
    cv2.imwrite("white_stripes.png", img)


if __name__ == "__main__":
    create_white_stripes_image(width=640, height=480, stripe_width=49)

