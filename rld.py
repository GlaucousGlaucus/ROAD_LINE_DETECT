import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coords(image, line_param):
    slope, intercept = line_param
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    lline = make_coords(image, left_fit_avg)
    rline = make_coords(image, right_fit_avg)
    return np.array([lline, rline])


def cannyyy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cannied = cv2.Canny(blurred, 50, 150)
    return cannied

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_intrest(image):
    height = image.shape[0]
    triangle = np.array([(200, height), (1100, height), (550, 250)])
    polygons = np.array([triangle])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# image = cv2.imread('road_image.jpg')
# lane_img = np.copy(image)
# cann = cannyyy(lane_img)
# cropped_image = region_of_intrest(cann)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# average_lines = average_slope_intercept(lane_img, lines)
# line_image = display_lines(lane_img, average_lines)
# final_image = cv2.addWeighted(lane_img, 0.8, line_image, 1, 1)
# cv2.imshow("result", final_image)
# cv2.waitKey(0)

# VID

cap = cv2.VideoCapture("road_video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    cann = cannyyy(frame)
    cropped_image = region_of_intrest(cann)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_lines)
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# To show the image with pixels to get dimensions for -> np.array([(200, height), (1100, height), (550, 250)])
#
# plt.imshow("result", cann)
# plt.waitKey(0)
