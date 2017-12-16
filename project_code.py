import glob
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from lesson_functions import *
import pickle

def train_classifier():
    '''
    Train and save a Linear support vector classifier
    '''
    cars = [image for image in glob.glob('training_data/vehicles/**/*.png', recursive=True)]
    notcars = [image for image in glob.glob('training_data/non-vehicles/**/*.png', recursive=True)]

    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC()
    svc.fit(X_train, y_train)

    dist_pickle = {}
    dist_pickle["svc"] = svc
    pickle.dump(dist_pickle, open( "classifier_pickle.p", "wb" ))

    return svc


def slide_multiple_windows(image, svc):
    '''
    Perform multiple sliding windows technic to detect rectangles with vehicles on image
    '''

    # y_start, y_stop, scale
    window_sizes = [(400, 464, 1.0),
                (420, 580, 1.5),
                (400, 660, 1.5),
                (400, 660, 2.0),
                (500, 660, 3),
                (464, 660, 3.5)]

    hot_windows = []
    for window_params in window_sizes:
        hot_windows += find_cars(image, window_params[0], window_params[1], window_params[2], svc)

    return hot_windows


seen_rectangles = []
def track_windows(rects):
    '''
    track detected rectangles with vehicles over video for smoothing
    '''
    global seen_rectangles
    seen_rectangles.append(rects)

    # track only fresh detected rectangles
    if len(seen_rectangles) > 10:
        seen_rectangles = seen_rectangles[len(seen_rectangles) - 10:]


def pipeline(image):
    '''
    whole pipeline to detect vehicles on an image
    '''
    # perform multiple sliding windows technic to detect rectangles with vehicles
    hot_windows = slide_multiple_windows(image, svc)

    # track detected rectangles for smoothing
    if len(hot_windows) > 0:
        track_windows(hot_windows)

    # apply heat threshold for removing false positives
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    for rect_set in seen_rectangles:
        heat = add_heat(heat, rect_set)
    heat = apply_threshold(heat, 1 + len(seen_rectangles) // 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    result_img, rects = draw_labeled_bboxes(np.copy(image), labels)
    return result_img

def read_classifier():
    '''
    load a trained SVC from a pickle
    '''
    return pickle.load( open( "classifier_pickle.p", "rb" ) )["svc"]


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(255, 0, 0), thick=5):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes

    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


if __name__ == '__main__':
    # svc = train_classifier()
    svc = read_classifier()

    # test_out_file = 'project_video_result.mp4'
    # clip_test = VideoFileClip('project_video.mp4')
    # clip_test_out = clip_test.fl_image(pipeline)
    # clip_test_out.write_videofile(test_out_file, audio=False)

    # draw_hog_examples(cars, notcars) 4 3 5
    img = mpimg.imread('test_images/test3.jpg')
    box_list = slide_multiple_windows(img, svc)
    box_list += box_list
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    heat = add_heat(heat,box_list)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, _ = draw_labeled_bboxes(np.copy(img), labels)

    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10,10))
    f.subplots_adjust(hspace = .4, wspace=.2)
    ax1.imshow(heatmap, cmap='hot')
    ax1.set_title('Heat Map', fontsize=14)

    ax2.imshow(draw_img)
    ax2.set_title('Detected Vehicle Positions', fontsize=14)
    plt.show()






