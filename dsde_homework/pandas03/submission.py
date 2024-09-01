import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic.csv) and answer the questions.
"""


def Q1(df):
    """
        Problem 1:
            How many rows are there in the titanic.csv?
            Hint: In this function, you must load your data into memory before executing any operations. To access titanic.csv, use the path /data/titanic.csv.
    """
    # df = pd.read_csv('/data/titanic.csv')
    return df.shape[0]


def Q2(df):
    '''
        Problem 2:
            Drop unqualified variables
            Drop variables with missing > 50%
            Drop categorical variables with flat values > 70% (variables with the same value in the same column)
            How many columns do we have left?
    '''
    def flat_value_ratio(df):
        def flat_value_ratio_(df,column):
            value_counts = df[column].value_counts()
            total = sum([v for k,v in value_counts.items()],0.0)
            max_item = max([v for k,v in value_counts.items()])
            return max_item/total
        return pd.Series({column:flat_value_ratio_(df,column) for column in df.columns})
    def isna_ratio(df):
        def ratio(df,column):
            total = len(df[column])
            return df[column].isna().sum() / total
        return pd.Series({column:ratio(df,column) for column in df.columns})

    is_object = lambda x : x == 'object'
    column_flat_value_more_than70 = lambda item : item[1] > 0.70
    flat_columns = [column[0] for column in filter(column_flat_value_more_than70,flat_value_ratio(df).items())]
    object_columns = [column[0]  for column in filter(lambda column : is_object(column[1]),df.dtypes.items())]
    flat_columns_to_remove = set(object_columns).intersection(flat_columns)
    missing_less_than_50_percent_columns = (isna_ratio(df) * 100 > 50) == False
    return len(df.loc[:,missing_less_than_50_percent_columns].drop(flat_columns_to_remove,axis=1).columns)


def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    
    return len(df[df['Survived'].isna() == False])


def Q4(df):
    '''
       Problem 4:
            Handle outliers
            For the variable “Fare”, replace outlier values with the boundary values
            If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
            If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
            What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    
    q1, q3 = (df['Fare'].quantile(q=0.25),df['Fare'].quantile(q=0.75))
    iqr = q3 - q1
    lower_bound = q1 - (1.5* iqr)
    upper_bound = q3 + (1.5* iqr)
    lower_outlier_to_lower_bound = lambda x: lower_bound if x < lower_bound else x
    upper_outlier_to_upper_bound = lambda x: upper_bound if x > upper_bound else x
    
    mean = df['Fare'].apply(lower_outlier_to_lower_bound).apply(upper_outlier_to_upper_bound).mean()
    return round(mean,2)


def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    
    imp = SimpleImputer(strategy='mean')
    mean = imp.fit_transform(df[['Age']]).mean()
    
    return round(mean,2)


def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    FEATURE = 'Embarked'
    
    encoder = OneHotEncoder(sparse_output=False,drop='first')
    encoded_embark = encoder.fit_transform(df[[FEATURE]])
    encoded_df = pd.DataFrame(encoded_embark,columns=encoder.get_feature_names_out([FEATURE]))
    mean = encoded_df['Embarked_Q'].mean()
    return round(mean,2)


def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection
    '''
    SURVIVED = 'Survived'
    df = df[df[SURVIVED].isna() == False]
    X = df.drop(SURVIVED, axis=1)
    y = df[SURVIVED]
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123,test_size=0.3,stratify=y,)
    result = y_train.value_counts()/y_train.shape[0]
    return int(result[1]*100)/100
