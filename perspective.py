import cv2
import numpy as np

image = cv2.imread(r"C:\Users\patma\OneDrive\Imagens\curso\Non_Affine"
".jpg")

gray  =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        paper_corners = approx
        break

if len(paper_corners) == 4:
    pts = paper_corners.reshape(4, 2)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype='float32')

        sorted_pts = pts[np.argsort(pts[:, 0])]

        left_pts = sorted_pts[:2]
        right_pts = sorted_pts[2:]

        rect[0] = left_pts[np.argmin(left_pts[:, 1])]
        rect[2] = left_pts[np.argmax(left_pts[:, 1])]
        rect[1] = right_pts[np.argmin(right_pts[:, 1])]
        rect[3] = right_pts[np.argmax(right_pts[:, 1])]

        return rect
    
    ordered_pts = order_points(pts)

    (tl, tr, bl, br) = ordered_pts
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    dst_pts = np.array([
        [0, 0], 
        [width-1, 0], 
        [0, height-1], 
        [width-1, height-1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))

    cv2.imshow('Original', image)
    cv2.imshow('Warp Perspective', warped)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

else:
    print('Não foi possível encontrar as coordenadas')