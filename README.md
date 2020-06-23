# Face_Detection_Viola_Jones

A face detection tool based on viola-jones machine learning algorithm

# How to run

Run (training): Train a cascaded classifier by following the given steps:
1. extract "train_posf" and "train_negf" files
1. Run "haar_features.py" to extract all possible haar feature values for each images in positive and negative face images
2. Run "threshold_optimization.py" to get optimized thresholds for each weak classifier(haar feature) 
3. Run "ada_boost.py" to get the most relevant weak classifiers(haar features), that give minimum error on classifying positive and negative face images, for faces in sequence
4. Run "cascade_training.py" to build a cascade out of the relevant weak classifiers extracted from adaboost. 


Run (for detection): Can use already trained cascade for detection purpose
1. Run "FaceDetection.py" followed by a space and <..relative directory to the images to be detected>.

# Please refer "Report.pdf" for implementation and project details
