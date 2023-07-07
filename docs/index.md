---
title: Breast Cancer Detection and Classification using Histopathological Images (Group 5)
layout: default
---
# Introduction
Breast cancer is a prevalent global cancer, causing significant mortality. In the United States alone, there were 284,200 cases and 44,130 deaths in 2021 [1]. Promptly and accurately identifying cancerous and benign breast tissues is vital for effective treatment. While histopathological data analyzed by trained medical professionals is currently used for classification, machine-learning techniques using medical imaging data, such as mammograms and ultrasound images, have shown promise in diagnosing breast cancer [2]. These techniques primarily distinguish between benign and malignant tumors, but further classification is necessary due to distinct subtypes that influence prognosis and treatment planning. The challenge lies in accurately identifying each subtype due to the complexity and variability of breast tissue characteristics, often relying on cellular measurements [3].

# Problem Statement
Despite advancements in breast cancer classification, there remains a challenge to achieving high accuracy and reliability. The motivation behind this project is to develop an ML model that not only accurately classifies breast tissues as cancerous or benign, but also classifies subtypes of each category &mdash; helping in further prognosis and diagnosis of the disease. 

# Dataset
We will utilize the publicly available Breast Cancer Histopathological Database [4]. The dataset is composed of 9,109 microscopic images of breast tumor tissue using different magnifying factors (40X, 100X, 200X, and 400X). It contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth). The benign and malignant samples are further classified into Adenosis, Fibroadenoma, Tubular Adenoma, Phyllodes Tumor for benign samples and Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma (Colloid), and Papillary Carcinoma for malignant samples.

# Methods
We aim to use unsupervised learning for exploratory analysis of our problem space, and supervised learning for classification.
<ol type='a'> 
<li> K-means and hierarchical clustering can identify common attributes within each disease subtype, aiding feature design and understanding. </li>
<li> Dimensionality reduction methods like PCA can improve tractability and feature prioritization. </li>
<li> With the potential for high dimensionality, a multi-class application of SVMs can help find optimal hyperplanes separating subtypes. </li>
<li> Decision trees can extract explicit if-then rules on feature values, making them suitable for cancer prediction. Random forests can lead to improved robustness &mdash; feature importances are valuable for interpretability. </li>
<li> With image data, convolutional/feed-forward neural networks could be viable to extract complex, non-linear relationships. </li>
</ol>

# Potential Results and Discussion
In this project, we expect to achieve several outcomes. Firstly, we will evaluate and compare our candidate machine learning models based on metrics like accuracy, precision/recall, F1 score, AUC-ROC, and the confusion matrix. This analysis will help us identify the most effective model for breast cancer classification. Given the healthcare context of our problem, false-positive rate is also a key consideration.

Secondly, we aim to attain high accuracy and reliability in distinguishing between cancerous and benign breast tissue to aid in timely diagnosis. We plan to calculate interpretability metrics like Shapley values to determine the most influential features, providing insights for medical professionals and researchers. We acknowledge the possibility of limitations and will explore additional techniques such as data augmentation, ensemble methods, multimodal classification, or advanced deep-learning architectures if applicable.

# Results and Discussion
Supervised Learning:
Support Vector Machine (SVM)

Wisconsin breast cancer dataset:

<img width="447" alt="image" src="https://github.com/CS7641-Group-5-Summer-2023/project-website/assets/78183814/283a56a5-b8d4-472c-b996-9592815cac26">
<img width="351" alt="image" src="https://github.com/CS7641-Group-5-Summer-2023/project-website/assets/78183814/5c41d1eb-22b9-461f-b68f-0558b73e67c5">

The above visualizations demonstrate the outstanding performance of Support Vector Machines (SVMs) when applied to the Wisconsin Breast Cancer dataset. It can achieve good performance for several reasons.

Firstly, SVMs excel at distinguishing between classes based on a feature set that allows for linear separation. The Wisconsin Breast Cancer dataset provides numerous useful features such as radius, texture, perimeter, area, and more. These features carry essential information of the condition, thereby allowing the SVM model to learn an effective decision boundary.

Secondly, SVMs are inherently proficient at binary classification tasks, which is the case with the breast cancer prediction problem. The ability to distinguish between two distinct classes enables the SVM model to optimize its performance on this dataset.

Moreover, the Wisconsin Breast Cancer dataset is well-balanced distribution and meticulous preprocessing. These attributes contribute towards making the data easier to separation via a hyperplane in the SVM's high-dimensional feature space.


Image-based dataset:

<img width="439" alt="image" src="https://github.com/CS7641-Group-5-Summer-2023/project-website/assets/78183814/57345e4c-137e-48f8-b1cb-eee3b6a3f0db">
<img width="344" alt="image" src="https://github.com/CS7641-Group-5-Summer-2023/project-website/assets/78183814/2787950a-e37b-4f4c-bec3-1afef41f2666">
  
The above visualizations demonstrate SVMs’ poor ability of handling image-based data. It isn’t suitable for such data for the following reason.

First, when using raw image data, each pixel in the image becomes a feature. This significantly increases the dimensionality of the feature space, which can lead to worse performance due to the curse of dimensionality.

Second, SVMs do not inherently handle translation, scale, and rotation invariance. This means that if an object (like a tumor) appears in different places in the image (translation), at different sizes (scale), or at different orientations (rotation), the SVM may not recognize it as the same object.

Last, SVMs do not automatically learn features from raw data. In image processing tasks, feature engineering (e.g., creating features that describe textures, shapes, or colors in the image) can help improve an SVM's performance. Without this kind of feature engineering, SVMs may not perform well on raw image data.

Unsupervised Learning:

K-means:
K-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.

Wisconsin breast cancer dataset:
Applying PCA to reduce dimensionality to 2 and visualization.
<img width="289" alt="image" src="https://github.com/CS7641-Group-5-Summer-2023/project-website/assets/67562398/e50e1eae-6e1b-4800-a333-c1f48a4de738">
Silhouette Score: 0.3845494883485513

A silhouette score of 0.3845 suggests that the clustering is reasonably effective, with some degree of separation and assignment correctness, but there may be room for improvement. 
K-means clustering may not be the most suitable method for breast cancer detection and classification. 

K-means clustering is primarily an unsupervised learning algorithm used for clustering data into groups based on similarity. It does not directly consider the class labels or target variable during training. 

For breast cancer detection and classification, we have labeled data where each sample is associated with a specific class (benign or malignant). In this case, supervised learning algorithms such as logistic regression, support vector machines (SVM), random forests, or neural networks are more commonly used for classification tasks. These algorithms take into account the labeled data and learn to classify new samples based on the patterns and relationships in the training data.


# References

> Siegel, R. L., Miller, K. D., Fuchs, H. E., & Jemal, A. (2021). Cancer statistics, 2021. CA: a cancer journal for clinicians, 71(1), 7-33

> Aksebzeci BH, Kayaalti Ö (2017) Computer-aided classification of breast cancer histopathological images. Paper presented at the 2017 Medical Technologies National Congress (TIPTEKNO)

> Murtaza, G., Shuib, L., Abdul Wahab, A.W. et al. Deep learning-based breast cancer classification through medical imaging modalities: state of the art and research challenges

> F. A. Spanhol, L. S. Oliveira, C. Petitjean and L. Heutte, "A Dataset for Breast Cancer Histopathological Image Classification," in IEEE Transactions on Biomedical Engineering, vol. 63, no. 7, pp. 1455-1462, July 2016, doi: 10.1109/TBME.2015.2496264.

> KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

> Support Vector Machines (SVMs): https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

> Decision Trees and Random Forests: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

> Neural Networks: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

> Convolutional Neural Networks: https://www.tensorflow.org/tutorials/images/cnn

# Contribution Table

<table rules="all">
  <thead>
    <tr>
      <th></th>
      <th style="text-align: center"> Ujani </th>
      <th style="text-align: center"> Xing </th>
      <th style="text-align: center"> Bobak </th>
      <th style="text-align: center"> Huijie </th>
      <th style="text-align: center"> Srihas </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="text-align: center">Introduction</th>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
    </tr>
    <tr>
      <th style="text-align: center">Problem Statement</th>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
    </tr>
    <tr>
      <th style="text-align: center">Dataset Description</th>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
    </tr>
    <tr>
      <th style="text-align: center">Methods</th>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
    </tr>
    <tr>
      <th style="text-align: center">Potential Results & Discussion</th>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
    </tr>
    <tr>
      <th style="text-align: center">Review and Editing</th>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
    </tr>
    <tr>
      <th style="text-align: center">Github Page</th>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
    </tr>
    <tr>
      <th style="text-align: center">Presentation Slides</th>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
    </tr>
        <tr>
      <th style="text-align: center">Recording</th>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1">*</font></td>
      <td style="text-align: center"><font size="+1"></font></td>
      <td style="text-align: center"><font size="+1"></font></td>
    </tr>
  </tbody>
</table>

Link to timeline chart - [Project Timeline](https://www.dropbox.com/s/8jzimta0l86ylog/ML%20Project%205%20Timeline.xlsx?dl=0)
