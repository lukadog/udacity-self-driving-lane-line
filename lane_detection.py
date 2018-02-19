# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import cv2
import math
from moviepy.editor import VideoFileClip


# load the file name of test images into test_images_list
test_images_list = os.listdir("test_images/")

# display the test images and processed images

fig = plt.figure()

# Mask the pavement images with lane colors: yellow and white  
def color_selection(img): 
    # white color selection
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_pixels = cv2.inRange(img, lower, upper)
    # yellow color selection
    lower = np.uint8([190, 190, 0])
    upper = np.uint8([255, 255, 255])
    yellow_pixels = cv2.inRange(img, lower, upper)
    # mask the image
    masked_pixels = cv2.bitwise_or(white_pixels, yellow_pixels)
    masked_image = cv2.bitwise_and(img, img, mask = masked_pixels)
    return masked_image

# Covert the RGB image into gray image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Apply gaussian blur filter
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Apply edge detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Mask region of interest
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Select vertices for the mask
def select_vertices(img):
    # first, define the polygon by vertices
    rows, cols = img.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return vertices

# Apply Hough transform to detect the line
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Draw lines on the image
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
def average_line(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (float(y2)-float(y1))/(float(x2)-float(x1))
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    # return two lanes in the format of slope and intercept
    return left_lane, right_lane 


def make_line_points(y1, y2, line):

    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))


def lane_points(img, left_lane, right_lane):
    lane_image = np.zeros_like(img)
    
    y1 = img.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle
    left_lane_points  = make_line_points(y1, y2, left_lane)
    right_lane_points  = make_line_points(y1, y2, right_lane)
    cv2.line(lane_image, left_lane_points[0], left_lane_points[1], color=[0, 255, 0], thickness=10)
    cv2.line(lane_image, right_lane_points[0], right_lane_points[1], color=[0, 255, 0], thickness=10)

    return lane_image
    

def process_image(image):
        
    color_filtered_image = color_selection(image)
    gray_image = grayscale(color_filtered_image)
    blurred_image = gaussian_blur(gray_image, kernel_size=15)
    edge_image = canny(blurred_image, low_threshold=50, high_threshold=150)
    vertices = select_vertices(edge_image)
    roi_image = region_of_interest(edge_image, vertices)
    lines = hough_lines(roi_image, rho=1, theta=np.pi/180, threshold=20, min_line_len = 20, max_line_gap = 300)
    left_lane, right_lane = average_line(lines)
    lane_image = lane_points(image, left_lane, right_lane)
    lane_image_scene = cv2.addWeighted(image, 1.0, lane_image, 0.50, 0.0)
    return lane_image_scene
    


# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 15.0, (540, 960))
# out = cv2.VideoWriter('test.avi', fourcc, 20.0, (540, 960))

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (540, 540))

# white_output = 'test_videos_output/solidWhiteRight.mp4'

# clip = VideoFileClip('test_videos/solidWhiteRight.mp4', video_input)
# new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
# new_clip.write_videofile(os.path.join('output_videos', video_output), audio=False)


# def process_video(video_input, video_output):

#     clip = VideoFileClip(os.path.join("test_videos", video_input))
#     processed = clip.fl_image(process_image)
#     processed.write_videofile(os.path.join("output_videos", video_output))

# process_video("solidWhiteRight.mp4", "white.mp4")    


cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')


while(cap.isOpened()):
    ret, frame = cap.read()
    RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_image = process_image(RGB_img)
    processed_image_RGB = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    out.write(processed_image_RGB)
    cv2.imshow('frame', processed_image_RGB)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



