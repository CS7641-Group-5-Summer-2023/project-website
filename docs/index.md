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

Supervised Learning:

1. Support Vector Machines (SVMs): SVMs are powerful models that can find an optimal hyperplane that separates different classes in a high-dimensional space. They can handle both linearly separable and non-linear cases using kernel functions.

2. Decision Trees and Random Forests:
A decision tree is a supervised learning algorithm used for classification and regression. It learns a series of explicit if-then rules on feature values to predict a target value. Random forests aggregate the predictions of many decision trees.

3. Neural Networks: 
Neural Networks (NNs) are algorithms used in supervised learning for tasks like classification and regression. They consist of interconnected layers of nodes or "neurons" that process input data and learn to make accurate predictions.

4. Convolutional Neural Networks:
Convolutional Neural Networks (CNNs) are a special type of Neural Network designed to process grid-like data such as images. They utilize convolutional layers, which automatically and adaptively learn spatial hierarchies of features. CNNs have been more efficient in tasks such as image classification and object detection.

5. Logistic Regression:
It is a simple and fast model, often used in binary classification problems. Despite its simplicity, it can perform well when the relationship between the features and the target variable is approximately linear.

# Potential Results and Discussion
In this project, we expect to achieve several outcomes. Firstly, we will evaluate and compare different machine learning models, including K-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, Neural Networks, Convolutional Neural Networks, and Logistic Regression, based on metrics like accuracy, precision, recall, F1 score, AUC-ROC, and the confusion matrix. This analysis will help us identify the most effective model for breast cancer classification. Secondly, our aim is to attain high accuracy and reliability in distinguishing between cancerous and benign breast tissues, which will contribute to timely detection and diagnosis of breast cancer. Additionally, we plan to calculate Shapley values to determine the most influential features driving the model's decisions, providing valuable insights for medical professionals and researchers. We acknowledge the possibility of limitations and will explore potential improvements, such as data augmentation, ensemble methods, or advanced deep learning architectures, if necessary. Throughout the research process, we will remain flexible and adaptive to refine our approach based on the insights gained from the experimental results.

# References

> Siegel, R. L., Miller, K. D., Fuchs, H. E., & Jemal, A. (2021). Cancer statistics, 2021. CA: a cancer journal for clinicians, 71(1), 7-33

> Aksebzeci BH, Kayaalti Ö (2017) Computer-aided classification of breast cancer histopathological images. Paper presented at the 2017 Medical Technologies National Congress (TIPTEKNO)

> Murtaza, G., Shuib, L., Abdul Wahab, A.W. et al. Deep learning-based breast cancer classification through medical imaging modalities: state of the art and research challenges

> Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions.” Advances in Neural Information Processing Systems. 2017

> Wisconsin Diagnostic Breast Cancer (WDBC) Dataset and Wisconsin Prognostic Breast Cancer (WPBC) Dataset.
http://ftp.ics.uci.edu/pub/machine-learning-databases/breast-cancer-wisconsin/

> K-Nearest Neighbors (KNN): https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

> Support Vector Machines (SVMs): https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

> Decision Trees and Random Forests: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

> Neural Networks: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

> Convolutional Neural Networks: https://www.tensorflow.org/tutorials/images/cnn

> Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

