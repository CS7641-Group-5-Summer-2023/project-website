---
title: Breast Cancer Detection (Group 5)
layout: default
---
# Introduction
Breast cancer is a prevalent global cancer, causing significant mortality. In the United States alone, there were 284,200 cases and 44,130 deaths in 2021 [1]. Timely and accurately identifying cancerous and benign breast tissues is vital for effective treatment. While histopathological data analyzed by trained medical professionals is currently used for classification, machine-learning techniques using medical imaging data, such as mammograms and ultrasound images, have shown promise in diagnosing breast cancer [2]. These techniques primarily distinguish between benign and malignant tumors, but further classification is necessary due to distinct subtypes that influence prognosis and treatment planning. The challenge lies in accurately identifying each subtype due to the complexity and variability of breast tissue characteristics, often relying on cellular measurements [3].

# Problem Statement
Despite the advancements in breast cancer classification, there still remains a challenge to achieving high accuracy and reliability. The motivation behind this project is to develop an ML model that not only accurately classifies breast tissues as cancerous or benign but also summarizes the most important features driving the decisions (a more game theory approach of calculating Shapely values for the cell features) [4]. 

# Dataset

We will utilize the publicly available Wisconsin Breast Cancer Diagnostic dataset [5], containing 357 samples labeled as "benign" and 212 samples labeled as "malignant." The dataset consists of features computed from digitized fine needle aspirate (FNA) images of breast masses, specifically describing cell nucleus characteristics. These features include mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each feature is represented by a real-valued value.


# Methods
We will use various models to solve this problem, including supervised learning and unsupervised learning. These are our candidates. 
a) K-means is a type of unsupervised learning algorithm used for clustering problems.
b) SVMs are powerful models that can find an optimal hyperplane that separates different classes in a high-dimensional space.
c) A decision tree is a supervised learning algorithm used for classification and regression. Random forests aggregate the predictions of many decision trees.
d) Neural Networks (NNs) are algorithms modeled after the human brain, which can be used in supervised learning for tasks like classification and regression. 
e) Convolutional Neural Networks (CNNs) are a special type of Neural Network designed to process grid-like data such as images.
f) Logistic Regression is a simple and fast model often used in binary classification problems.

# Potential Results and Discussion
In this project, we expect to achieve several outcomes. Firstly, we will evaluate and compare different machine learning models, including K-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, Neural Networks, Convolutional Neural Networks, and Logistic Regression, based on metrics like accuracy, precision, recall, F1 score, AUC-ROC, and the confusion matrix. This analysis will help us identify the most effective model for breast cancer classification. Secondly, our aim is to attain high accuracy and reliability in distinguishing between cancerous and benign breast tissues, which will contribute to timely detection and diagnosis of breast cancer. Additionally, we plan to calculate Shapley values to determine the most influential features driving the model's decisions, providing valuable insights for medical professionals and researchers. We acknowledge the possibility of limitations and will explore potential improvements, such as data augmentation, ensemble methods, or advanced deep learning architectures, if necessary. Throughout the research process, we will remain flexible and adaptive to refine our approach based on the insights gained from the experimental results.

# References

> Siegel, R. L., Miller, K. D., Fuchs, H. E., & Jemal, A. (2021). Cancer statistics, 2021. CA: a cancer journal for clinicians, 71(1), 7-33

> Aksebzeci BH, Kayaalti Ö (2017) Computer-aided classification of breast cancer histopathological images. Paper presented at the 2017 Medical Technologies National Congress (TIPTEKNO)

> Murtaza, G., Shuib, L., Abdul Wahab, A.W. et al. Deep learning-based breast cancer classification through medical imaging modalities: state of the art and research challenges

> Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions.” Advances in Neural Information Processing Systems. 2017

> Wisconsin Diagnostic Breast Cancer (WDBC) Dataset and Wisconsin Prognostic Breast Cancer (WPBC) Dataset.
http://ftp.ics.uci.edu/pub/machine-learning-databases/breast-cancer-wisconsin/

> KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

> Support Vector Machines (SVMs): https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

> Decision Trees and Random Forests: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

> Neural Networks: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

> Convolutional Neural Networks: https://www.tensorflow.org/tutorials/images/cnn

> Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

