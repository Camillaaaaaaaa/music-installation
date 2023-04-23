import cv2
import numpy as np

# You should replace these 3 lines with the output in calibration step
DIM = (1280, 720)
K = np.array(
    [[514.5414424836712, 0.0, 653.4964962297564], [0.0, 512.971045355613, 349.06466799474686], [0.0, 0.0, 1.0]])
D = np.array([[-0.055518374762623585], [-0.09175821441414458], [0.03752987112976255], [-0.011492161564419939]])


def undistort(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    undistort("WIN_20230414_14_20_37_Pro.jpg")

