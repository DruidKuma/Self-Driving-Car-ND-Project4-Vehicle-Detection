## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use a trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_example_1.png
[image3]: ./output_images/hog_example_2.png
[image4]: ./output_images/sliding_windows.png
[image5]: ./output_images/detection_result_1.png
[image6]: ./output_images/detection_result_2.png
[image7]: ./output_images/detection_result_3.png
[image8]: ./output_images/heat_final_1.png
[image9]: ./output_images/heat_final_2.png
[image10]: ./output_images/heat_final_3.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for ALL channels (combined) of the image:

![][image2]
![][image3]

The code for extracting the HOG attributes can be found in `lesson_functions.py` (lines 8-13 and lines 40-43)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. It was pretty hard to find the best set, cause even slight difference lead either to detecting a huge number of false positives or to very low detection rate. the final decision for HOG parameters was `YCrCb` color space, `9` orient, `8` pixels per cell, `2` cells per block, using all color channels for gradient.
I also tried previously to play with `LUV` color space, and the results were slightly poor. Interesting thing, though, that for parameters with `LUV` color space (`11` orient, `8` pixels per cell, `2` cells per block) the accuracy for the trained SVC was always lower than for my final choice parameters (on average 93% to 98-99% accordingly). Actually, this fact was one of the rest to switch to another color space and experiment with it.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training a classifier is situated in `project_code.py` file (lines 9-33).
I have used provided labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.

Some interesting things that I have discovered for myself during training and tuning different parameters:

* I have tried alignining evenly the numbers of cars and not-cars images for training, but when cleaning up the code I decided to try leave all training data as is and found out that it had no negative impact on the result classifier. I guess that this is because the difference between number of car samples and non-car samples is pretty small (rougly 8600 against 8900 images)
* I refused using Scaler for the input data as using it significantly decreased my detection rate.
* For fitting the classifier I've used only HOG attributes (described above), but for prediction I also extracted bin spatial and color histogram attributes (`lesson_functions.py` lines 137-138). Using these attributes while training always gave me low detection rate, but ignoring them on predictions gave much more false positives. I found this a bit strange and think that it was me mistaken somewhere. This place would have been the first point for improvements.

The final SVC has acurracy of about 98%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I define the region of interest between `400` and `660` pixels on the y axis of the image and defined `6` different set of parameters for multiple window sliding search. The idea was that on the top part of the image we can see only small vehicle images (as they are far away), and a big windows won't detect a vehicle there. From top to bottom the size of potential vehicle became bigger (as the vehicle become closer) and small sliding windows could potentialy miss them. So I started with small windows on the top and scaled them on the way to bottom. 

The region and scale of the windows:

| Y start  | Y stop | Scale |
|:--------:|:------:|:-----:|
| 400      | 464    | 1     |
| 420      | 580    | 1.5   |
| 400      | 660    | 1.5   |
| 400      | 660    | 2     |
| 500      | 660    | 3     |
| 464      | 660    | 3.5   |

All sliding windows looks like this:
![][image4]

The code for defining windows is located in file `project_code.py` (lines 42-47). The sliding window algorithm is located in `lesson_functions.py` (lines 84-152)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on provided above scales using YCrCb 3-channel HOG features, which provided a satisfactory result. This approach still detects slightly more false positives than I wanted, but we'll deal with them later  Here are some example images:

![][image5]
![][image6]
![][image7]

To optimize the performance I decided to keep track of the previously detected rectangles. This resulted in smoother rectangles between frames and improved the overall performance on video stream (when classifier hasn't detected clearly a vehicle, previous history of detections could be used). To keep the history small, I have tracked maximum of 10 last detections. The code for this can be found in `project_code.py` (lines 56-66).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code is located in `project_code.py` (lines 80-91) and `lesson_functions.py` (lines 50-81)  

Here are examples showing the heatmaps and the bounding boxes, then overlaid on the last frame of video, for the examples shown above:

![][image8]
![][image9]
![][image10]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* First of all I would improve the extracting features technics (as was described above). I am pretty sure that combination of HOG with color histograms and bin spatials would do a better thing with a certain set of hyperparameters.
* With current implementation the model might fail when there would be much noise or shadows, or there would be many vehicles on the road (like in a traffic jam)
* Current implementation sometimes detects vehicles on the opposite side of the road
* Another way for improvement would be smart choice of region of interest. Instead of hard-coding the region on y axis, it should be computed (as y-axis constraints would differ, for example, if the car would drive upwards or downwards the road). Also it would be much better to restrct the region also for x-axis (to avoid detections on the opposite side of the road and to adjust to different road widths)
* Personally I believe that a sort of neural network would do even better thing here, than simpler classifiers like SVM.

