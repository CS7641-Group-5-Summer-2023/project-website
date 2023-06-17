---
title: Breast Cancer Detection (Group 5)
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
