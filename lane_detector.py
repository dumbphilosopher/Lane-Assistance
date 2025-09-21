import cv2
import numpy as np

def canny_edge_detector(image):
    """
    Applies the Canny edge detection algorithm to an image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    return canny_image

def region_of_interest(image):
    """
    Applies an image mask to isolate the region of interest (the road).
    """
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def create_coordinates(image, line_parameters):
    """
    Calculates the (x, y) coordinates for a line based on its slope and intercept.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Averages the detected line segments to find the single left and right lane lines.
    This version is robust against frames where no lines are detected for a lane
    and filters out nearly-horizontal lines to prevent large coordinate errors.
    """
    left_fit = []
    right_fit = []
    
    if lines is None:
        return np.array([]) # Return an empty array

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        # --- THIS IS THE FIX ---
        # We filter out lines that are too horizontal by setting a slope threshold.
        slope_threshold = 0.5 
        if abs(slope) < slope_threshold:
            continue # Skip this line if it's too horizontal

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    averaged_lines = []
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = create_coordinates(image, left_fit_average)
        averaged_lines.append(left_line)
        
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = create_coordinates(image, right_fit_average)
        averaged_lines.append(right_line)
    
    return np.array(averaged_lines)

def display_lines(image, lines):
    """
    Draws lines on a black image.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# --- Main Execution ---
cap = cv2.VideoCapture("test_video.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    canny_image = canny_edge_detector(frame)
    cropped_image = region_of_interest(canny_image)
    
    lines = cv2.HoughLinesP(cropped_image, 
                            rho=2,
                            theta=np.pi/180,
                            threshold=100,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=5)
    
    averaged_lines = average_slope_intercept(frame, lines)
    
    line_image = display_lines(frame, averaged_lines)
    
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    cv2.imshow("Lane Assistance", combo_image)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()