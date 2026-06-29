# Purchase Prediction - Logistic Regression Model

A machine learning project using **Logistic Regression** to predict whether a customer will purchase a product based on their age and estimated salary.

## 📋 Project Overview

This project implements a binary classification model that predicts customer purchase behavior from social network advertising data. The model uses user demographics (age and estimated salary) to classify whether a customer made a purchase (1) or did not make a purchase (0).

### Dataset
- **Name**: Social Network Ads
- **Size**: 400 samples
- **Features**: 2 (Age, EstimatedSalary)
- **Target**: 1 binary class (Purchased: 0 or 1)
- **Train/Test Split**: 75% / 25%

## 🎯 Project Goals

1. Build a logistic regression classifier for binary purchase prediction
2. Preprocess and scale features for optimal model performance
3. Evaluate model accuracy and generate confusion matrix
4. Demonstrate complete ML pipeline: data loading → preprocessing → training → evaluation

## 🛠️ Technologies & Libraries

- **Python 3**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models and tools
  - `train_test_split` - Data splitting
  - `StandardScaler` - Feature scaling
  - `LogisticRegression` - Classification model
  - `confusion_matrix` - Model evaluation
  - `accuracy_score` - Performance metrics
- **Matplotlib** - Data visualization
- **Google Colab** - Development environment

## 📊 Model Pipeline

### 1. **Data Loading**
   - Import the Social_Network_Ads.csv dataset
   - Extract features (Age, EstimatedSalary) and target (Purchased)

### 2. **Data Splitting**
   - Split data into training (75%) and test (25%) sets
   - Random state = 0 for reproducibility

### 3. **Feature Scaling**
   - Apply StandardScaler to normalize features
   - Mean = 0, Standard Deviation = 1
   - Essential for logistic regression performance

### 4. **Model Training**
   - Algorithm: Logistic Regression
   - Random state = 0 for reproducibility
   - Training set size: 300 samples

### 5. **Model Evaluation**
   - Confusion Matrix: True positives, True negatives, False positives, False negatives
   - Accuracy Score: Overall model performance percentage

## 🚀 How to Use

### Run in Google Colab
```
Click the Colab badge in the notebook to run directly in your browser
```

### Run Locally
```python
# 1. Clone the repository
git clone https://github.com/Rebexcode/Purchase-Prediction-Logistic-Regression-Model.git

# 2. Install dependencies
pip install numpy pandas scikit-learn matplotlib

# 3. Run the notebook
jupyter notebook purchase_prediction_logistic_regression_model.ipynb
```

### Required Dataset
- Place `Social_Network_Ads.csv` in the same directory as the notebook
- Dataset columns: `Age`, `EstimatedSalary`, `Purchased`

## 📈 Model Performance

The model is evaluated using:

- **Accuracy Score**: Percentage of correct predictions on test set
- **Confusion Matrix**: Breakdown of classification results:
  ```
  TN  FP
  FN  TP
  ```
  - True Negatives (TN): Correctly predicted no purchase
  - False Positives (FP): Incorrectly predicted purchase
  - False Negatives (FN): Incorrectly predicted no purchase
  - True Positives (TP): Correctly predicted purchase

## 📁 Project Structure

```
Purchase-Prediction-Logistic-Regression-Model/
├── README.md
├── purchase_prediction_logistic_regression_model.ipynb
└── Social_Network_Ads.csv
```

## 💡 Key Concepts

### Logistic Regression
- Supervised learning algorithm for binary classification
- Uses sigmoid function to map predictions to probability (0-1)
- Outputs probability that a sample belongs to class 1

### Feature Scaling
- Standardizes features to have mean 0 and standard deviation 1
- Improves model convergence and performance
- Prevents features with larger scales from dominating

### Train-Test Split
- 75% training data: Used to fit the model
- 25% test data: Used for unbiased performance evaluation
- Prevents overfitting and provides realistic accuracy estimate

## 🔍 Model Insights

The model learns relationships between:
- **Age** - Customer age in years
- **EstimatedSalary** - Estimated annual salary
- **Purchase Likelihood** - Probability of making a purchase

The decision boundary is learned during training to optimally separate customers who purchased from those who didn't.

## ✨ Future Improvements

- Add additional features (e.g., gender, location, user ID)
- Implement cross-validation for robust performance estimation
- Add visualization of decision boundaries
- Test alternative algorithms (Random Forest, SVM, Neural Networks)
- Perform hyperparameter tuning
- Add ROC-AUC curve analysis
- Implement ROC curve and precision-recall metrics

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Rebexcode** - [GitHub Profile](https://github.com/Rebexcode)

## 🙏 Acknowledgments

- Scikit-learn documentation and tutorials
- Google Colab for computational resources
- Machine Learning community for best practices

---

**Note**: This is an educational project demonstrating fundamental machine learning concepts using a simple binary classification task.
