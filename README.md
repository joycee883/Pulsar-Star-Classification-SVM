# 🎯 Project Spotlight: Pulsar Star Classification with Machine Learning

### 📋 Project Overview
The Pulsar Star Classification project is designed to accurately classify pulsar stars from other astronomical bodies using advanced machine learning techniques. By leveraging various statistical features from pulsar signal data, this project aims to enhance the understanding of these unique celestial objects and improve classification accuracy in astrophysics. The project encompasses end-to-end data handling, model training, evaluation, and an interactive web-based interface for real-time predictions using Streamlit.

### 🛠️ Tools Used
Programming Language: Python <br>
Libraries: <br>
* NumPy: For efficient numerical computations. <br>
* Pandas: For handling and preprocessing data. <br>
* Scikit-learn: For model development, training, and evaluation. <br>
* Matplotlib & Seaborn: For data visualization and exploratory analysis. <br>
Streamlit: For creating an interactive and user-friendly web app. <br>

### 🔍 Key Steps
1. Data Collection & Preprocessing <br>
* Data Source: A dataset containing features like mean, standard deviation, skewness, and kurtosis of pulsar signals. <br>
* Data Cleaning: Checked for missing values and outliers, removing or imputing data where necessary. <br>
* Feature Engineering: Focused on statistical features critical for pulsar classification, ensuring high-quality input for model training. <br>

2. Exploratory Data Analysis (EDA) <br>
* Conducted visual analysis using scatter plots and histograms to understand the distributions and relationships between features. <br>
* Visualized correlations using heatmaps to confirm the importance of various features in distinguishing pulsar signals from noise. <br>

3. Declare Feature Vector and Target Variable <br>
* Specified the feature vector (input variables) and target variable (pulsar vs. non-pulsar). <br>

4. Split Data into Separate Training and Test Set <br>
* Divided the dataset into training and testing subsets to evaluate model performance. <br>

5. Feature Scaling <br>
* Applied feature scaling techniques to ensure consistent ranges for model inputs, enhancing performance. <br>

6. Run SVM with Default Hyperparameters <br>
* Implemented the SVM model using default settings, achieving an accuracy score of 0.9827. <br>

7. Run SVM with Different Kernels <br>
RBF Kernel: <br>
* With C=100.0: Accuracy of 0.9832. <br>
* With C=1000.0: Accuracy of 0.9816. <br>
Linear Kernel: <br>
* With C=1.0: Accuracy of 0.9830. <br>
* With C=100.0: Accuracy of 0.9832. <br>
* With C=1000.0: Accuracy of 0.9832. <br>
Polynomial Kernel: <br>
* With C=1.0: Accuracy of 0.9807. <br>
Sigmoid Kernel: <br>
* With C=1.0: Accuracy of 0.8858. <br>
* With C=100.0: Accuracy of 0.8855. <br>

8. Model Evaluation <br>
Confusion Matrix: <br>
|              | Predicted No (0) | Predicted Yes (1) |
|--------------|-------------------|-------------------|
| Actual No (0)|      3289         |        17         |
| Actual Yes (1)|       44         |        230        |
 
True Positives (TP): 3289 <br>
True Negatives (TN): 230 <br>
False Positives (FP): 17 <br>
False Negatives (FN): 44 <br>

GridSearch CV Best Score: 0.9793 <br>
Best Parameters: <br>
{'C': 10, 'gamma': 0.3, 'kernel': 'rbf'} <br>
* Chosen Estimator : SVC(C=10, gamma=0.3) <br>

### 📊 Key Findings
Model Results: <br>
The SVM model with an RBF kernel and optimal parameters (C=10, gamma=0.3) provided the highest accuracy, achieving 0.9830 on the test set. <br>
The model demonstrated strong classification performance, particularly in identifying pulsar signals with high precision and recall values. <br>

### 🌐 Application
The Pulsar Star Classification app, built using Streamlit, allows users to input statistical features from pulsar signals to predict whether they are pulsar stars. The app is designed with a clean, intuitive interface, providing real-time predictions. This tool can be particularly useful for: <br>
* Astronomy Enthusiasts: Learning about pulsar classification and the unique properties of these stars. <br>
* Researchers: Quickly analyzing data and making predictions based on statistical features without the need for extensive computational resources. <br>

### 🔮 Future Improvements
* Incorporating additional features such as temporal data from pulsar signals to enhance classification accuracy. <br>
* Expanding the dataset with diverse astronomical observations for more generalized predictions. <br>
* Adding advanced visualizations to help users understand the classification process and model performance. <br>


Discover the Pulsar Star Classification app! This Streamlit application leverages data science to help users classify pulsar stars based on their statistical features.
Visit the live app here: [Link to your app].
