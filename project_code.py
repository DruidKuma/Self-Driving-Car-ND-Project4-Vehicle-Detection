import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from skimage.feature import hog


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	'''
	Takes an image, a list of bounding boxes, 
	and optional color tuple and line thickness as inputs
	then draws boxes in that color on the output
	'''
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img


def find_matches(img, template_list):
	'''
	Takes an image and a list of templates as inputs
	then searches the image and returns the a list of bounding boxes 
	for matched templates
	'''
    # Make a copy of the image to draw on
    imcopy = np.copy(img)
    # Define an empty list to take bbox coords
    bbox_list = []

    method = cv2.TM_CCOEFF_NORMED
    for template in template_list:
        tmp_img = mpimg.imread(template)
        result = cv2.matchTemplate(img, tmp_img, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        w, h = (tmp_img.shape[1], tmp_img.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        bbox_list.append((top_left, bottom_right))
    return bbox_list


def color_hist(img, nbins=32, bins_range=(0, 256)):
	'''
	Compute color histogram features  
	'''
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


def plot_color_hist(rh, gh, bh, bincen):
	'''
	Plot a figure with all three bar charts (color histogram)
	'''
	if rh is not None:
	    fig = plt.figure(figsize=(12,3))
	    plt.subplot(131)
	    plt.bar(bincen, rh[0])
	    plt.xlim(0, 256)
	    plt.title('R Histogram')
	    plt.subplot(132)
	    plt.bar(bincen, gh[0])
	    plt.xlim(0, 256)
	    plt.title('G Histogram')
	    plt.subplot(133)
	    plt.bar(bincen, bh[0])
	    plt.xlim(0, 256)
	    plt.title('B Histogram')
	    fig.tight_layout()
	else:
	    print('Your function is returning None for at least one variable...')


def bin_spatial(img, color_space='RGB', size=(32, 32)):
	'''
	Compute color histogram features  
	Pass the color_space flag as 3-letter all caps string
	like 'HSV' or 'LUV' etc.
	'''
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features


def data_look(car_list, notcar_list):
	'''
	Return some characteristics of the dataset 
	'''
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	'''
	Return HOG features and visualization
	'''
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
	'''
	Extract features from a list of images
	Have this function call bin_spatial() and color_hist()
	'''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img_name in imgs:
        # Read in each one by one
        image = mpimg.imread(img_name)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)  
        
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        rhist, ghist, bhist, bin_centers, hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


if name == '__main__':
	image = mpimg.imread('bbox-example-image.jpg')
	# image = mpimg.imread('temp-matching-example-2.jpg')
	templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
            'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']

    rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

	bboxes = find_matches(image, templist)
	result = draw_boxes(image, bboxes)
	plt.imshow(result)


	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# Define HOG parameters
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	# Call our function with vis=True to see an image output
	features, hog_image = get_hog_features(gray, orient, 
	                        pix_per_cell, cell_per_block, 
	                        vis=True, feature_vec=False)


	# Plot the examples
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(image, cmap='gray')
	plt.title('Example Car Image')
	plt.subplot(122)
	plt.imshow(hog_image, cmap='gray')
	plt.title('HOG Visualization')





	images = glob.glob('*.jpeg')
	cars = []
	notcars = []
	for image in images:
	    if 'image' in image or 'extra' in image:
	        notcars.append(image)
	    else:
	        cars.append(image)
	        
	car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
	                        hist_bins=32, hist_range=(0, 256))
	notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
	                        hist_bins=32, hist_range=(0, 256))

	if len(car_features) > 0:
	    # Create an array stack of feature vectors
	    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	    # Fit a per-column scaler
	    X_scaler = StandardScaler().fit(X)
	    # Apply the scaler to X
	    scaled_X = X_scaler.transform(X)
	    car_ind = np.random.randint(0, len(cars))
	    # Plot an example of raw and scaled features
	    fig = plt.figure(figsize=(12,4))
	    plt.subplot(131)
	    plt.imshow(mpimg.imread(cars[car_ind]))
	    plt.title('Original Image')
	    plt.subplot(132)
	    plt.plot(X[car_ind])
	    plt.title('Raw Features')
	    plt.subplot(133)
	    plt.plot(scaled_X[car_ind])
	    plt.title('Normalized Features')
	    fig.tight_layout()
	else: 
	    print('Your function only returns empty feature vectors...')