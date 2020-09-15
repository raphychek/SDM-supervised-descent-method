This folder contains a code to solve face landkmars inspired by the article "Supervised Descent Method and its Applications to Face Alignment" by 
X. Xiong et al.

# How to use it?

## What's needed to try it out
- A training dataset with images (.jpg), landmarks (.pts), boundig box around the face (.mat).
- A test dataset with images (.jpg), landmarks (.pts), boundig box around the face (.mat).

Training and tests was done using the Hellen Dataset (https://ibug.doc.ic.ac.uk/download/annotations/helen.zip).

## Advised organisation 
By default, it would be easier if folders are organised as follows:

.SDM.py
.helen/
    .testset/ : all test jpgs and pts
    .trainset/ : all train jpgs and pts
.bboxed/ : train set and test set .mat files with bounding box
.predictions/ : folder where to save predictions on the test set

After training, if default parameters are not given, the weights will by in .weights.mat


## Arguments 
The only required argument is the mode ("train" or "test"). Others are not needed if the organisation is as advised.

## Training
For a custom training, one needs:

`python --mode="train" --weights_path=WEIGHT_PATH --train_bbox=.MAT_PATH --train_images=JPG_PATH+".jpg" --train_lms=.PTS_PATH+"*pts" --train_epochs=NUM_EPOCHS`

## Testing
For a custom testing, one needs:

`python --mode="test" --weights_path=WEIGHT_PATH --test_bbox=.MAT_PATH --test_images=JPG_PATH+".jpg" --test_lms=.PTS_PATH+"*pts" --output_dir=OUTPUT_PATH` 


# Notes

Simple linear regression gave absurd landmarks. Lasso was tried but too slow and not optimal (designed to find a laplacian distribution, here the distribution is closer to normal). We chose to stick with Ridge regression.

HoG descriptor was difficult to chose. We decided to stick with opencv implementation, as it allows to chose localisations on which to focus. Implementation for sklearn.image did not.
We changed the parameters in order to have a smaller descriptor vector, for computational purposes. We kepts to a limit for which the perception and MAE of the predictions seems not much degraded. Smaller vector means less features, and potentially less information to learn (and infer) on, so less good results.

Implementation was kind of straightforward, but with some challenge on the choice of regressions and descriptor choice and parameters.

This implementation shall only work with a real test dataset for which we know the .pts ground truth. This is easy to change it in order to use it for classical inference, but it seemed not to be the point of this work.

To have better estimations by having all landmarks points in the image, there is a function called "big_bbox" which enlarges the bbox by a certain ratio to give as a parameter. We noticed better results when using it with a little ratio such as 5-10% but we did not really studied it. To stick to the article, the default ratio parameter is 0.
