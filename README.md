# Facebook Comment Volume Prediction Project
Predicting the number of comments on a Facebook post

## Overview
This project aims to predict the number of comments on Facebook posts using the "Facebook Comment Volume Dataset." The dataset consists of various features related to the Facebook pages and posts, with the target variable being the number of comments a post receives. The goal is to understand which factors influence post engagement and how different machine learning models perform in predicting comment volumes.

## Dataset
https://archive.ics.uci.edu/dataset/363/facebook+comment+volume+dataset

## Dataset Description

### Features
1\. **Page Popularity/Likes:** Defines the popularity or support for the source of the document.  
2\. **Page Checkins:** Describes how many individuals have visited the place associated with the page.  
3\. **Page Talking About:** Defines the daily interest of individuals towards the source of the document/post.  
4\. **Page Category:** Defines the category of the source of the document, such as place, institution, brand, etc.  
5-29\. **Derived Features:** Aggregated features calculated by page, including min, max, average, median, and standard deviation of essential features.  
30-34\. **CC1 to CC5:** Essential features representing the number of comments at different time intervals.  
35\. **Base Time:** Selected time to simulate the scenario.  
36\. **Post Length:** Character count in the post.  
37\. **Post Share Count:** Counts the number of shares the post received.  
38\. **Post Promotion Status:** Binary encoding indicating whether the post is promoted (1) or not (0).  
39\. **H Local:** Describes the hours for which we have the target variable/comments received.  
40-46\. **Post Published Weekday:** Binary encoding for the day (Sunday...Saturday) on which the post was published.  
47-53\. **Base DateTime Weekday:** Binary encoding for the day (Sunday...Saturday) on the selected base Date/Time.  

### Target Variable
- **Target Variable:** The number of comments in the next H hours (H is given in Feature no 39).

## Exploratory Data Analysis (EDA) and Visualizations
- **Summary Statistics:** Displayed basic statistics for numerical features.
- **Missing Values:** Examined and visualized missing values in the dataset.
- **Duplicates:** Checked for and handled duplicate values.
- **Data Distributions:** Visualized the distribution of the target variable and key features.
- **Correlation Matrix:** Explored the correlation between features using a heatmap.
- **Pair Plots:** Investigated relationships between features and the target variable.
- **Boxplots:** Created boxplots to identify outliers in numerical columns.

## Machine Learning Models
Implemented and evaluated the performance of various machine learning models using the scikit-learn library. The models include:
1. **Linear Regression**
2. **Random Forest Regressor**
3. **Support Vector Regressor (SVM)**
4. **Gradient Boosting Regressor**

### Data Preprocessing
- **Standardization:** Scaled numerical features using StandardScaler.
- **Encoding:** Used OrdinalEncoder for ordinal encoding and handled categorical features.
- **Train-Test Split:** Split the dataset into training and testing sets.

### Model Evaluation
- **Baseline Models:** Evaluated mean and median models as baselines.
- **Grid Search:** Utilized GridSearchCV to find the best hyperparameters for each model.
- **Performance Metrics:** Calculated and compared Root Mean Squared Error (RMSE) for model evaluation.

## Results
- **Linear Regression:** Applied to understand the basic linear relationship.
- **Random Forest Regressor:** Achieved an RMSE of 15.71 on the full dataset.
- **Support Vector Regressor (SVM):** Explored different hyperparameter combinations.
- **Gradient Boosting Regressor:** Experimented with varying learning rates and the number of estimators.

## How to Use
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook to explore the analysis and model development.
