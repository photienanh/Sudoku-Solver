import cv2
import numpy as np

def reorder(points):
    points = points.reshape((4, 2))
    corners = np.zeros((4, 2), dtype=np.float32)

    # Tổng x + y nhỏ nhất => điểm góc trên trái
    # Tổng x + y lớn nhất => điểm góc dưới phải
    add = points.sum(1)
    corners[0] = points[np.argmin(add)]
    corners[2] = points[np.argmax(add)]

    # Hiệu x - y nhỏ nhất => góc trên phải
    # Hiệu x - y lớn nhất => góc dưới trái
    diff = np.diff(points, axis=1)
    corners[1] = points[np.argmin(diff)]
    corners[3] = points[np.argmax(diff)]

    return corners

def detect_board(img):
    if type(img) == str:
        gray_img = cv2.imread(img, 0)
    else:
        gray_img = img
    
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area
    if biggest is None:
        return None

    # Sắp xếp điểm và warp ảnh
    original_corners = reorder(biggest)
    side = 450  # Kích thước ảnh sau khi chuẩn hóa
    warped_corners = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(original_corners, warped_corners)
    warp = cv2.warpPerspective(gray_img, matrix, (side, side))

    return warp

def split_cells(warped_img, size=450):
    cells = []
    cell_size = size // 9
    for y in range(9):
        for x in range(9):
            x1 = x * cell_size
            y1 = y * cell_size
            cell = warped_img[y1:y1 + cell_size, x1:x1 + cell_size]
            cells.append(cell)
    return cells
