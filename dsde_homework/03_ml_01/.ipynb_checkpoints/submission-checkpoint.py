import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
class MushroomClassifier:
    def __init__(self, data_path): # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

    def Q1(self): # DO NOT modify this line
        """
            1. (From step 1) Before doing the data prep., how many "na" are there in "gill-size" variables?
        """
        df = self.df
        return df['gill-size'].isna().sum()

    def Q2_(self):
        df = self.df
        columns_to_remove = [
            'id',
            'gill-attachment',
            'gill-spacing',
            'gill-size',
            'gill-color-rate',
            'stalk-root',
            'stalk-surface-above-ring',
            'stalk-surface-below-ring',
            'stalk-color-above-ring-rate',
            'stalk-color-below-ring-rate',
            'veil-color-rate',
            'veil-type',
        ]
        def drop_label_na(df):
            return df[df['label'].notna()]
        def remove_columns(df,columns_to_remove):
            return df.drop(columns=columns_to_remove)
        df =  (df
               .pipe(drop_label_na)
               .pipe(remove_columns,columns_to_remove)
                   )
        return df

    def Q2(self): # DO NOT modify this line
        """
            2. (From step 2-4) How many rows of data, how many variables?
            - Drop rows where the target (label) variable is missing.
            - Drop the following variables:
            'id','gill-attachment', 'gill-spacing', 'gill-size','gill-color-rate','stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring-rate','stalk-color-below-ring-rate','veil-color-rate','veil-type'
            - Examine the number of rows, the number of digits, and whether any are missing.
        """
        return self.Q2_().shape

    def Q3_(self):
        df = self.Q2_()
        def label_encoder(df,column):
            encode_map = {'e':1,'p':0}
            result = df.copy()
            result[column] = result[column].map(encode_map).astype(int)
            return result

        label_column = ['label']
        categorial_columns = df.iloc[:,1:-1].columns
        numerical_columns = ['cap-color-rate']
        preprocessor = ColumnTransformer(
            transformers=[
                ('label_encoding',Pipeline(steps=[
                    ('label_encoder',FunctionTransformer(func=lambda df : label_encoder(df, 'label'), validate=False)),
                ]),label_column),
                ('categorial_imputation',Pipeline(steps=[
                    ('mode_imputer',SimpleImputer(strategy='most_frequent')),
                ]),categorial_columns),
                ('numerical-imputation',Pipeline(steps=[
                    ('mean_imputer',SimpleImputer(strategy='mean'))
                ]),numerical_columns),
            ],
            remainder='passthrough',
        )
        final_pipeline = Pipeline(steps=[
            ('drop_label_na',FunctionTransformer(func=lambda df : df.dropna(subset=label_column))),
            ('preprocess',preprocessor),
        ])
        final_pipeline.fit(df)
        preprocessed_dataframe = pd.DataFrame(final_pipeline.transform(df),columns=df.columns)
        return preprocessed_dataframe
        

    def Q3(self): # DO NOT modify this line
        """
            3. (From step 5-6) Answer the quantity class0:class1
            - Fill missing values by adding the mean for numeric variables and the mode for nominal variables.
            - Convert the label variable e (edible) to 1 and p (poisonous) to 0 and check the quantity. class0: class1
        """
        return self.Q3_().label.value_counts()


    def Q4_(self): # DO NOT modify this line
        """
            4. (From step 7-8) How much is each training and testing sets
            - Convert the nominal variable to numeric using a dummy code with drop_first = True.
            - Split train/test with 20% test, stratify, and seed = 2020.
        """
        preprocessed_dataframe = self.Q3_()
        categorial_columns = preprocessed_dataframe.iloc[:,1:-1].columns
        one_hot_encode_pipeline = ColumnTransformer(
            transformers=[
                ('onehot',OneHotEncoder(drop='first',sparse_output=False),categorial_columns)
            ],
            remainder='passthrough',
            force_int_remainder_cols=False,
        )
        X_features = preprocessed_dataframe.drop('label',axis=1)
        y = preprocessed_dataframe['label'].astype(int)
        one_hot_encode_pipeline.fit(X_features)
        encoded_columns = one_hot_encode_pipeline.named_transformers_['onehot'].get_feature_names_out()
        X = one_hot_encode_pipeline.transform(X_features)
        
        X_train,X_test, y_train,y_test =train_test_split(X,y,stratify=y,test_size=0.2,random_state=2020)
        return X_train,X_test, y_train,y_test 

    def Q4(self):
        X_train,X_test, y_train,y_test = self.Q4_()
        return X_train.shape, X_test.shape
        


    def Q5_(self):
        """
            5. (From step 9) Best params after doing random forest grid search.
            Create a Random Forest with GridSearch on training data with 5 CV with n_jobs=-1.
            - 'criterion':['gini','entropy']
            - 'max_depth': [2,3]
            - 'min_samples_leaf':[2,5]
            - 'N_estimators':[100]
            - 'random_state': 2020
        """
        X_train,X_test, y_train,y_test = self.Q4_()
        clf = RandomForestClassifier()
        param_grid = {
            'criterion':['gini','entropy'],
            'max_depth': [2,3],
            'min_samples_leaf':[2,5],
            'n_estimators':[100],
            'random_state':[2020],
        }
        grid_search = GridSearchCV(param_grid=param_grid,estimator=clf,cv=5,n_jobs=-1)
        grid_search.fit(X_train,y_train)
        return grid_search
        
    def Q5(self):
        grid_search = self.Q5_()
        return grid_search.best_params_


    def Q6(self):
        """
            5. (From step 10) What is the value of macro f1 (Beware digit !)
            Predict the testing data set with confusion_matrix and classification_report,
            using scientific rounding (less than 0.5 dropped, more than 0.5 then increased)
        """
        X_train,X_test, y_train,y_test = self.Q4_()
        grid_search = self.Q5_()
        report = classification_report(y_test,grid_search.predict(X_test))
        return report

    def pipelining(self):
        df = self.df
