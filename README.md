# 🚢 Titanic Survival Prediction — Machine Learning

A Machine Learning project that predicts whether a passenger survived the Titanic disaster based on passenger information such as age, gender, passenger class, fare, and family size.

The project follows the complete Machine Learning workflow, from data cleaning and exploratory data analysis (EDA) to preprocessing, model training, and evaluation.

---

## 📌 Project Overview

The Titanic dataset is a classic binary classification problem.

The goal is to build a Machine Learning model that predicts the `Survived` status of a passenger.

### Target Variable

- `0` → Did not survive
- `1` → Survived

Two Machine Learning algorithms are implemented and compared:

- Logistic Regression
- Random Forest Classifier

---

## 🎯 Objectives

- Understand and clean the Titanic dataset
- Handle missing values
- Perform Exploratory Data Analysis (EDA)
- Create meaningful features
- Encode categorical variables
- Split the dataset into training and testing sets
- Scale features where required
- Train multiple Machine Learning models
- Compare model performance
- Analyze training and testing accuracy
- Identify potential overfitting

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / VS Code
- Git & GitHub

---

## 📂 Project Structure

```text
Titanic-ML/
│
├── titanic.ipynb
├── train.csv
├── titanic_updated.csv
└── README.md
```

---

## 🔄 ML Workflow
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison

---

## 🧹 Data Cleaning

The dataset was examined for:

- Missing values
- Duplicate records
- Incorrect data types
- Unnecessary columns
- Missing Values

The following approach was used:

- Age → Missing values replaced using the median
- Embarked → Missing values replaced using the mode
- Cabin → Removed due to a large number of missing values

---

## 📊 Exploratory Data Analysis

Several visualizations were created to understand the relationships between passenger characteristics and survival.

- EDA Performed
- Survival distribution
- Survival by gender
- Survival by passenger class
- Age distribution
- Fare distribution
- Age vs survival
- Fare vs survival
- Family relationships
- Correlation analysis

Examples of questions explored:

- Did females have a higher survival rate?
- Did passenger class affect survival?
- Did age influence survival?
- Did family size affect survival?

---

## ⚙️ Feature Engineering

A new feature called FamilySize was created:

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

Where:

- SibSp → Number of siblings/spouses aboard
- Parch → Number of parents/children aboard
- +1 → Includes the passenger themselves

An IsAlone feature was also created:

df["IsAlone"] = np.where(
    df["FamilySize"] == 1,
    1,
    0
)

This identifies whether the passenger was traveling alone.

---

## 🔢 Data Preprocessing
Categorical Encoding

Categorical variables such as Sex and Embarked were converted into numerical values using One-Hot Encoding.

df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)

Feature and Target Separation
X = df.drop("Survived", axis=1)
y = df["Survived"]
Train-Test Split

The dataset was divided into:

- 80% Training data
- 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
Feature Scaling

Standardization was applied for Logistic Regression:

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Random Forest does not require feature scaling because it is a tree-based algorithm.

---

## 🤖 Machine Learning Models
1. Logistic Regression

Logistic Regression was used as a baseline classification model.

log_model = LogisticRegression(max_iter=1000)

log_model.fit(
    X_train_scaled,
    y_train
)
2. Random Forest Classifier

Random Forest was used as a non-linear ensemble model.

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

---

## 📈 Model Evaluation

The models were evaluated using:

- Training Accuracy
- Testing Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Training Accuracy Comparison

The training accuracy of both models was visualized using Matplotlib.

Testing Accuracy Comparison

The testing accuracy of both models was also compared to determine how well each model generalizes to unseen data.

---

## 🔍 Overfitting Analysis

Training and testing accuracy were compared for both models.

A large difference between training and testing accuracy can indicate overfitting.

For example:

Training Accuracy  → 99%
Testing Accuracy   → 82%

This indicates that the model performs extremely well on training data but considerably worse on unseen data.

Random Forest can sometimes show a larger training-testing gap because of its ability to fit complex patterns.

---

## 📊 Model Comparison

The final models are compared based on their testing performance.

Model	Training Accuracy	Testing Accuracy
Logistic Regression	80	81
Random Forest	85	82

The model with the better testing performance and a reasonable training-testing gap can be selected as the preferred model.

---

## 💡 Key Learnings

Through this project, the following Machine Learning concepts were explored:

- Data cleaning
- Missing value treatment
- Exploratory Data Analysis
- Feature engineering
- Categorical encoding
- Feature scaling
- Train-test splitting
- Binary classification
- Logistic Regression
- Random Forest
- Model evaluation
- Confusion matrices
- Overfitting analysis
- Model comparison

---

## 🚀 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- Feature selection
- Comparing additional classification algorithms
- Using a preprocessing Pipeline
- Handling the Cabin feature more effectively
- Extracting information such as passenger titles from Name
- Deploying the final model using Streamlit
- Creating a web interface for survival prediction

---

## 📚 Dataset

This project uses the Titanic dataset, a commonly used dataset for learning binary classification and Machine Learning workflows.

---

## 👩‍💻 Author

Chetna Manral

Machine Learning Project — Titanic Survival Prediction

---

## ⭐ Conclusion

This project demonstrates a complete Machine Learning workflow for a binary classification problem in titanic dataset.

Logistic Regression provides a simple and interpretable baseline, while Random Forest can capture more complex relationships between features.

The final model should be selected based on testing performance and generalization, rather than training accuracy alone.
