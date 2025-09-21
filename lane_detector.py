import cv2
import numpy as np

def canny_edge_detector(image):
    """
    Applies the Canny edge detection algorithm to an image.

    Args:
        image: The input image.

    Returns:
        An image with Canny edges detected.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise and smooth edges
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Canny edge detection
    canny_image = cv2.Canny(blurred_image, 50, 150)
    
    return canny_image

def region_of_interest(image):
    """
    Applies an image mask to isolate the region of interest (the road).

    Args:
        image: The input image (should be the Canny edge detected image).

    Returns:
        The masked image, showing edges only within the defined polygon.
    """
    height = image.shape[0]
    # Define the vertices of the polygon (trapezoid)
    # This will need to be adjusted based on the camera's perspective
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    
    # Create a black mask with the same dimensions as the image
    mask = np.zeros_like(image)
    
    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, polygons, 255)
    
    # Apply the mask to the image using a bitwise AND operation
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def create_coordinates(image, line_parameters):
    """
    Calculates the (x, y) coordinates for a line based on its slope and intercept.
    The line will span from the bottom of the image to about 3/5ths of the way up.

    Args:
        image: The original image, used to get dimensions.
        line_parameters: A tuple containing the slope and intercept of the line.

    Returns:
        An array of [x1, y1, x2, y2] representing the line's coordinates.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * (3/5)) # A point towards the middle of the image
    
    # Calculate the x-coordinates based on the y-coordinates and the line equation (y = mx + b -> x = (y-b)/m)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Averages the detected line segments to find the single left and right lane lines.

    Args:
        image: The original image.
        lines: The output from the Hough Transform.

    Returns:
        An array containing the coordinates of the averaged left and right lines.
    """
    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Fit a 1st degree polynomial (a line) to the points and get the slope and intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Lines with a negative slope are on the left, positive slope are on the right
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    # Average the slopes and intercepts for each side
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    
    # Calculate the coordinates for the left and right lines
    left_line = create_coordinates(image, left_fit_average)
    right_line = create_coordinates(image, right_fit_average)
    
    return np.array([left_line, right_line])


def display_lines(image, lines):
    """
    Draws lines on a black image.

    Args:
        image: The original image (used for dimensions).
        lines: The lines to be drawn.

    Returns:
        An image with the detected lines drawn on it.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draw the line on the black image
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # Blue line, 10px thickness
    return line_image

# --- Main Execution ---

# Open the video file
cap = cv2.VideoCapture("test_video.mp4")

while(cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (end of stream?). Exiting ...")
        break
    
    # 1. Apply Canny Edge Detection
    canny_image = canny_edge_detector(frame)
    
    # 2. Isolate the Region of Interest
    cropped_image = region_of_interest(canny_image)
    
    # 3. Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_image, 
                            rho=2,              # Distance resolution in pixels
                            theta=np.pi/180,    # Angle resolution in radians
                            threshold=100,      # Minimum number of votes (intersections in Hough grid cell)
                            lines=np.array([]), # Minimum number of pixels making up a line
                            minLineLength=40,   # Maximum gap in pixels between connectable line segments
                            maxLineGap=5)
    
    # 4. Average the detected lines to get single left and right lanes
    averaged_lines = average_slope_intercept(frame, lines)
    
    # 5. Create an image with the detected lines drawn on it
    line_image = display_lines(frame, averaged_lines)
    
    # 6. Combine the line image with the original frame to overlay the lines
    # The weights (0.8 and 1) can be adjusted for desired transparency
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    # 7. Display the result
    cv2.imshow("Lane Assistance", combo_image)
    
    # Wait for a key press and exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()