# HCPTaskfMRI
Goal of this project is to test different neural network architecture to classify HCP task fMRI data.

### Dataset
The dataset consists of Task fMRI data for 1042 subjects for 7 different tasks where each task consist of several sub-tasks. There are a total of 21 sub-tasks across all the tasks.

* EMOTION
* GAMBLING
* MOTOR
* WM
* LANGUAGE
* SOCIAL
* RELATION

#### Train Test Split
The dataset is split into 60% training and 40% test data. The training data is further split into 80% training and 20% validation dataset.

### Model
#### CNN model
This is a 1d-CNN model with 6 layers of 1-d CNN kernel interleaved by max pooling layer after every 2 CNN layer. Finally it has 2 layer of MLP at the end for classification.
#### Result
```
         class  precision   recall   f1-score   support

           0       0.84      0.87      0.85     12180
           1       0.82      0.86      0.84      8120
           2       0.63      0.50      0.56     18676
           3       0.52      0.70      0.60     18676
           4       0.89      0.88      0.89     36134
           5       0.88      0.91      0.89     30855
           6       0.20      0.08      0.11      1624
           7       0.37      0.60      0.46      1624
           8       0.00      0.00      0.00       812
           9       0.70      0.67      0.69      8526
          10       0.63      0.67      0.65      8526
          11       0.91      0.90      0.91     19488
          12       0.88      0.87      0.87     12992
          13       0.81      0.73      0.77      9338
          14       0.85      0.85      0.85      9338
          15       0.85      0.80      0.82      9338
          16       0.83      0.84      0.84      9338
          17       0.83      0.83      0.83      9338
          18       0.90      0.87      0.88      9338
          19       0.86      0.83      0.84      9338
          20       0.86      0.82      0.84      9338

    accuracy                           0.80    252937
   macro avg       0.72      0.72      0.71    252937
weighted avg       0.80      0.80      0.80    252937
```
