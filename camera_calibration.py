import cv2
import numpy as np
import glob
import os

def calibrate_camera(images_folder, pattern_size=(9,6), square_size=25.0):
    """
    Calibrate camera using chessboard pattern.
    Returns the intrinsic matrix (camera matrix).
    """
    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) scaled by square_size
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Load all jpg images from the folder
    images = glob.glob(os.path.join(images_folder, '*.jpg'))

    if not images:
        print("No images found!")
        return None, None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(300)

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Calibration successful.")
    print("Intrinsic matrix:\n", mtx)
    return mtx, dist


def save_intrinsic_matrix(mtx, filename='intrinsic.npy'):
    """
    Save the intrinsic matrix to a .npy file
    """
    np.save(filename, mtx)
    print(f"Intrinsic matrix saved to {filename}")


def load_intrinsic_matrix(filename='intrinsic.npy'):
    """
    Load the intrinsic matrix from a .npy file
    """
    mtx = np.load(filename)
    print(f"Loaded intrinsic matrix from {filename}:\n", mtx)
    return mtx


# Example usage:
if __name__ == "__main__":
    # Replace this with your actual image folder
    image_folder = 'Chessboard_image'

    mtx, dist = calibrate_camera(image_folder)


    if mtx is not None:
        save_intrinsic_matrix(mtx)
        # Optional: load it back to test
        _ = load_intrinsic_matrix()
