---
title: Breast Cancer Detection (Group 5)
layout: default
---
# Introduction
Breast cancer is one of the most common cancers worldwide, with 284,200 breast cancer cases and 44,130 deaths in 2021 in the US alone [1]. Early detection and accurate classification of breast tissues as either cancerous or benign are crucial for timely and effective treatment. While these classifications are currently performed by trained medical professionals from histopathological data, over the years, researchers have made significant progress in developing machine-learning techniques to diagnose breast cancer. These techniques leverage features from medical imaging data, such as mammograms and ultrasound images from breast cancer patients [2].  Generally, the classification distinguishes breast tumors into two types, benign and malignant. Benign is a noninvasive (non-cancerous) while malignant is an invasive (cancerous) type of tumor. But, both tumors have further subtypes that must be diagnosed individually because each may lead to different prognoses and treatment plans. The current problem lies in the variability and complexity of breast tissue characteristics, making accurate classification a non-trivial task. These subtypes are often detected by cellular measurements, such as the cell's radius, cell texture, the shape of nuclei, etc., and proper diagnosis requires accurate identification of each subcategory of breast cancer [3].

# Problem Statement
Despite the advancements in breast cancer classification, there still remains a challenge to achieving high accuracy and reliability. The motivation behind this project is to develop an ML model that not only accurately classifies breast tissues as cancerous or benign but also summarizes the most important features driving the decisions (a more game theory approach of calculating Shapely values for the cell features) [4]. 

# Dataset

We will use the publicly available Wisconsin Breast Cancer Diagnostic dataset [5], with samples labeled “benign” (357 samples) or “malignant” (212 samples), The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe the characteristics of the cell nuclei in the image. The 10 real-valued features computed for each cell nucleus are:
a) radius (mean of distances from the center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)


# Methods
We will use various models to solve this problem, including supervised learning and unsupervised learning. These are our candidate models.

Unsupervised Learning:

1. K-Nearest Neighbors (KNN): KNN is a type of instance-based learning which provides useful predictions in medical diagnosis problems. Even though KNN can be used for supervised learning tasks, we will use it in an unsupervised context as a clustering algorithm.
<br>https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans<br>

Supervised Learning:

1. Support Vector Machines (SVMs): SVMs are powerful models that can find an optimal hyperplane that separates different classes in a high-dimensional space. They can handle both linearly separable and non-linear cases using kernel functions.
<br>https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html<br>

2. Decision Trees and Random Forests:
A decision tree is a supervised learning algorithm used for classification and regression. It learns a series of explicit if-then rules on feature values to predict a target value. Random forests aggregate the predictions of many decision trees.
<br>https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier<br>

3. Neural Networks: 
Neural Networks (NNs) are algorithms used in supervised learning for tasks like classification and regression. They consist of interconnected layers of nodes or "neurons" that process input data and learn to make accurate predictions.
<br>https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier<br>

4. Convolutional Neural Networks:
Convolutional Neural Networks (CNNs) are a special type of Neural Network designed to process grid-like data such as images. They utilize convolutional layers, which automatically and adaptively learn spatial hierarchies of features. CNNs have been more efficient in tasks such as image classification and object detection.
<br>https://www.tensorflow.org/tutorials/images/cnn<br>

5. Logistic Regression:
It is a simple and fast model, often used in binary classification problems. Despite its simplicity, it can perform well when the relationship between the features and the target variable is approximately linear.
<br>https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression<br>


# Potential Results and Discussion
In this project, our goal is to develop a machine learning model that accurately classifies breast tissues as cancerous or benign using the Wisconsin Breast Cancer Diagnostic dataset. Additionally, we aim to summarize the most important features driving the model's decisions by employing a game theory approach to calculate Shapley values for the cell features.

To evaluate the performance of our models, we will utilize various quantitative metrics commonly used in machine learning. Some potential metrics we plan to use include:

1. Accuracy: It measures the overall correctness of the model's predictions, representing the ratio of correctly classified samples to the total number of samples.

2. Precision: It indicates the proportion of correctly classified malignant samples among the samples predicted as malignant. A higher precision value reflects fewer false positives.

3. Recall (Sensitivity): It represents the proportion of correctly classified malignant samples out of the total actual malignant samples. A higher recall value signifies fewer false negatives.

4. F1 Score: It is the harmonic mean of precision and recall, providing a balanced measure between the two. It is particularly useful when the dataset is imbalanced.

5. Area Under the ROC Curve (AUC-ROC): It measures the trade-off between the true positive rate (sensitivity) and the false positive rate. A higher AUC-ROC value indicates better discrimination between the two classes.

6. Confusion Matrix: It presents a tabular representation of the model's predictions compared to the ground truth, showing true positives, true negatives, false positives, and false negatives.

The potential results of our project could include:

1. Evaluation and Comparison of Models: We will train and evaluate different models, including K-Nearest Neighbors (KNN), Support Vector Machines (SVMs), Decision Trees, Random Forests, Neural Networks (NNs), Convolutional Neural Networks (CNNs), and Logistic Regression. We will compare their performance based on the aforementioned metrics to identify the most effective model for breast cancer classification.

2. Accuracy and Reliability: We aim to achieve high accuracy and reliability in classifying breast tissues as cancerous or benign. By leveraging various machine learning algorithms, we expect to improve upon the existing methods and contribute to the early detection and diagnosis of breast cancer.

3. Feature Importance and Interpretability: Through the game theory approach of calculating Shapley values, we intend to identify and summarize the most important features that drive the model's decisions. This can provide valuable insights to medical professionals and researchers, helping them understand the underlying factors contributing to the classification results.

4. Potential Limitations and Future Directions: It is important to acknowledge that the results of the project may vary during the research process. Factors such as data quality, model selection, and hyperparameter tuning can influence the outcomes. If the initial results are not satisfactory, we will explore further improvements, such as data augmentation techniques, ensemble methods, or more advanced deep learning architectures.

By evaluating and discussing the potential results of our machine learning project, we can set clear expectations and provide a roadmap for further investigation and refinement. It is essential to remain flexible and adaptive throughout the research process, allowing for adjustments and improvements based on the insights gained from the experimental results.

# References

> Siegel, R. L., Miller, K. D., Fuchs, H. E., & Jemal, A. (2021). Cancer statistics, 2021. CA: a cancer journal for clinicians, 71(1), 7-33

> Aksebzeci BH, Kayaalti Ö (2017) Computer-aided classification of breast cancer histopathological images. Paper presented at the 2017 Medical Technologies National Congress (TIPTEKNO)

> Murtaza, G., Shuib, L., Abdul Wahab, A.W. et al. Deep learning-based breast cancer classification through medical imaging modalities: state of the art and research challenges

> Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions.” Advances in Neural Information Processing Systems. 2017

> Wisconsin Diagnostic Breast Cancer (WDBC) Dataset and Wisconsin Prognostic Breast Cancer (WPBC) Dataset.
http://ftp.ics.uci.edu/pub/machine-learning-databases/breast-cancer-wisconsin/
