import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn .preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

class exam:
    def __init__(self, data: pd.DataFrame): # DO NOT modify this line
        self.data = data
        self.X_train = None
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 

    def Q1(self): # DO NOT modify this line
        """
        1. Data Preparation and Feature Selection
           In this step, select and prepare a subset of relevant features for predicting Titanic survival outcomes. 
           Specifically:
           - Select the target variable 'survived' and the following features:
             'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'adult_male', and 'alone'.
           - Handle missing values by filling them appropriately (e.g., median for numerical columns like 'age' and 'fare',
             and mode for categorical columns like 'embarked').
           - Convert categorical variables to numerical values using one-hot encoding with "drop the first category in each feature" so that they can be used in the model.

           This step ensures the data is complete, properly formatted, and ready for machine learning.

           **Remarks:**
           1. You must return the shape of the modified dataset to confirm changes were made correctly (return `self.data.shape`).
           2. Remember to update `self.data` as it will be used in subsequent methods.
        """
        df = self.data
        feature_columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'adult_male', 'alone']
        target_column = 'survived'
        selected_columns = feature_columns + [target_column]
        df = df.loc[:,selected_columns]
        numerical_columns = ['age', 'fare']
        median_imputer = SimpleImputer(strategy='median')
        for col in numerical_columns:
            df[col] = pd.DataFrame(median_imputer.fit_transform(df[[col]]))

        # categorical_columns = ['pclass','sex', 'sibsp','parch','embarked', 'class']
        categorical_columns = ['sex','embarked','class']
        mode_imputer = SimpleImputer(strategy='most_frequent')
        for col in categorical_columns:
            df[col] = pd.DataFrame(mode_imputer.fit_transform(df[[col]]))

        # Convert categorical valirables to numerical values using one-hot encoding with drop first
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        df = pd.concat([df.drop(columns=categorical_columns),encoded_df], axis=1)

        self.data = df
        return self.data.shape

    def Q2(self): # DO NOT modify this line
        """
        2. Splitting and Standardizing the Data
           Now that the data is ready, split it into training and testing sets to evaluate the model’s performance later:
           - Separate the target variable ('survived') from the selected features.
           - Use a 70-30 split for training and testing data, setting `random_state=2024` for reproducibility, 
             and enable stratification to maintain class distribution in the split.
           - Calculate and return the proportion of survivors in the training set as a quick metric to check data balance.
           
           By the end of this step, you will have the datasets ready for model training and evaluation.

           **Remarks:**
           1. Return the survivability rate in the training set, rounded to two decimal places (`round(y_train.mean(), 2)`), 
          which shows the proportion of survivors.
           2. Update `self.X_train`, `self.X_test`, `self.y_train`, and `self.y_test` for use in the final question.
           **Hint**
           You can use a try-except statement.
        """
        df = self.data
        numerical_columns = ['age', 'fare']
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        # feature_columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'adult_male', 'alone']
        target_column = 'survived'
        X = df.drop(columns=[target_column])
        y = df[target_column]
        RANDOM_STATE= 2024

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=RANDOM_STATE)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = self.y_test

        return round(y_train.mean(), 2)

    def Q3(self):
        """
        3. Model Training and Prediction
        Using logistic regression, train a predictive model to determine if a passenger survived:
        - Initialize a Logistic Regression model with `max_iter=10000` to ensure the model converges.
        - Train (fit) the model on the training dataset.
        - Use the trained model to make predictions on the test dataset and return the first prediction to verify output format.

        This step completes the machine learning pipeline, allowing you to assess model accuracy by comparing predictions 
        to actual test outcomes.

        **Remarks:**
        1. Return the first element of the prediction array (`y_pred[0]`) to verify that the model outputs correctly.
        """
        max_iter=10_000
        lr = LogisticRegression(max_iter=max_iter)
        lr.fit(X=self.X_train, y=self.y_train)
        y_pred = lr.predict(self.X_test)
        return y_pred[0]
