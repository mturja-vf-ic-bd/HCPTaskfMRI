# HCPTaskfMRI
Goal of this project is to test different neural network architecture to classify HCP task fMRI data.

### Dataset
The dataset comprises task fMRI data for 1042 subjects, encompassing 7 distinct tasks. Each task comprises multiple sub-tasks, totaling 21 sub-tasks across all tasks.

* Emotion
   * non-emo
   * fear-emo
* GAMBLING
   * loss-gamb
   * win-gamb
* MOTOR
   * hand-mot
   * foot-mot
   * tongue-mot
* WORKING MEMORY
   * tool2b_wm
   * body0b_wm
   * face2b_wm
   * tool0b_wm
   * body2b_wm
   * place2b_wm
   * face0b_wm
   * place0b_wm
* LANGUAGE
   * story-lang
   * math-lang
* SOCIAL
   * mental-soc
   * random-soc
* RELATION
   * relat-reson
   * match-reson

#### Train Test Split
The dataset is partitioned into two subsets: 60% for training and 40% for testing. Within the training subset, an 80-20 split is applied, resulting in 80% designated for training data and 20% for validation data.

### Model
#### CNN model
The model architecture consists of a 1D Convolutional Neural Network (1D-CNN) comprising six layers of 1D CNN kernels. These CNN layers are alternated with Dropout, max-pooling layers and ReLU, with a max-pooling layer inserted after every two consecutive CNN layers. Finally, the model concludes with two layers of Multi-Layer Perceptron (MLP) for classification purposes.
#### Result
```
         class  precision   recall   f1-score   support

      non-emo       0.84      0.87      0.85     12180
     fear-emo       0.82      0.86      0.84      8120
    loss-gamb       0.63      0.50      0.56     18676
     win-gamb       0.52      0.70      0.60     18676
   story-lang       0.89      0.88      0.89     36134
    math-lang       0.88      0.91      0.89     30855
     hand-mot       0.20      0.08      0.11      1624
     foot-mot       0.37      0.60      0.46      1624
   tongue-mot       0.00      0.00      0.00       812
  relat-reson       0.70      0.67      0.69      8526
  match-reson       0.63      0.67      0.65      8526
   mental-soc       0.91      0.90      0.91     19488
   random-soc       0.88      0.87      0.87     12992
    tool2b_wm       0.81      0.73      0.77      9338
    body0b_wm       0.85      0.85      0.85      9338
    face2b_wm       0.85      0.80      0.82      9338
    tool0b_wm       0.83      0.84      0.84      9338
    body2b_wm       0.83      0.83      0.83      9338
   place2b_wm       0.90      0.87      0.88      9338
    face0b_wm       0.86      0.83      0.84      9338
   place0b_wm       0.86      0.82      0.84      9338

    accuracy                           0.80    252937
   macro avg       0.72      0.72      0.71    252937
weighted avg       0.80      0.80      0.80    252937
```
