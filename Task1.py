
#REFERENCE LINKS
#https://www.youtube.com/watch?v=drp_mr2x6A8 


import cv2
import numpy as np

# function to load the images
def load_image(file_path):
    try:
        return cv2.imread(file_path)
    except Exception as e:
        print(f"Error loading image from {file_path}: {str(e)}")
        return None

# function for geometrical transformation
def perspective_transform(image, board, src_points, dst_points):
    # gets the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # transformation done
    transformed_image = cv2.warpPerspective(image, matrix, (board.shape[1], board.shape[0]))

    return transformed_image

def main():
    image = load_image('image.png')
    board = load_image('board.jpg')

    if image is None or board is None:
        return

# gets the coordinate of the images that must be replaced or overlapped
    tl = (0, 0)
    tr = (780, 0)
    bl = (0, 380)
    br = (780, 380)

    tl2 = (143, 195)
    tr2 = (584, 42)
    bl2 = (118, 404)
    br2 = (619, 275)

    # gets the coordinates of board and images
    src_points = np.float32([tl, tr, bl, br])
    dst_points = np.float32([tl2, tr2, bl2, br2])

    # apply the transformation matrix
    transformed_image = perspective_transform(image, board, src_points, dst_points)

    # combined two images
    combined_image = cv2.addWeighted(board, 1, transformed_image, 0.9, 0)

    # display final image
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
