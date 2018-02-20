# **Finding Lane Lines on the Road** 

## Project Summary

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/color_selection.png 
[image2]: ./examples/blurred_image.png 
[image3]: ./examples/edge_image.png 
[image4]: ./examples/grayscale.png 
[image5]: ./examples/lane_image.png
[image6]: ./examples/lane_line.png 
[image7]: ./examples/ROI_image.png 
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

Step 1: Color filter image.

The color selection is performed to remove the background that is in different color from the lanes.

![alt text][image1]

Step 2: Convert the RGB image into grayscale.

![alt text](https://github.com/lukadog/udacity-self-driving-lane-line/tree/master/examples/color_selection.png  "Grayscale")

Grayscale is performed to convert the color image to grayscale image. This is to prepare for the edge detection in subsequent steps.

Step 3: Blur the image.

Use Gaussian kernel to blur the image, this is to remove the sharp noises that may affect the edge detection result.

Step 4: Edge detection.

Use Canny kernel to detect the edges. Those edges can be later on used for line detection.

Step 5: ROI image selection.

Only keep the bottom part of the image where the pavement is usually be so the noises outside the pavement are not considered in the hough transform.

Step 6. Hough transform

Hough tranform is performed to detect the straight lines.

Step 7. Lane detection by averaging the detected lines.

Toom many lines might be detected in the hough transform, an average operation was performed to reduce the line into two: left line and right line.

Step 8. Draw the lines on blank image with green color and thickness = 10

Step 9. Overlay the line image with original image to show the final result





converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


[[[ 96 513]
  [384 324]
  [576 324]
  [864 513]]]

[[[532 335 852 513]]

 [[606 382 834 513]]

 [[527 338 641 403]]

 [[294 461 419 356]]

 [[282 460 416 355]]

 [[282 459 413 356]]

 [[531 334 575 358]]

 [[293 461 418 356]]

 [[534 341 829 511]]]
[ -0.81202204 693.98557126]
[ 0.56747388 36.20927803]




### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
