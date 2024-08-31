import json
import os


def read_env():
    try :
        with open('.env.local') as f:
            for line in f:
                if not line.startswith('#') and not line.isspace():
                    key, value = line.strip().split('=')
                    os.environ[key] = value
    except FileNotFoundError:
        pass
read_env()
def is_debug():
    return os.getenv('DEBUG', False) == 'true'
    

import json

import pandas as pd

# get_first :: Series s => s a -> a
get_first = lambda xs : xs.iloc[0]
# select_where:: (Row -> bool) -> DataFrame -> DataFrame
select_where = lambda condition: lambda df : df[df.apply(condition, axis=1)]

path = ''
if is_debug() :
    path = 'data/Usvideos.csv'
    json_path = 'data/US_category_id.json'
else:
    path = "/data/GBvideos.csv"
    json_path = '/data/GB_category_id.json'

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from GB (GBvideos.csv and GB_category_id.json) and answer the questions.
"""


def Q1():
    """
    1. How many rows are there in the GBvideos.csv after removing duplications?
    - To access 'GBvideos.csv', use the path '/data/GBvideos.csv'.
    """
    return pd.read_csv(path).drop_duplicates().shape[0]


def Q2(vdo_df):
    """
    2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    def have_more_dislikes_than_likes(rows):
        return 1 if (rows['dislikes'] > rows['likes']).sum() > 0 else 0
    return vdo_df.groupby('title').apply(lambda rows : have_more_dislikes_than_likes(rows)).sum()
    # select_where(lambda row: row['dislikes']  > row['likes']) (vdo_df.groupby('title').agg({
    #     'title' : get_first,
    #     'likes':'sum',
    #     'dislikes':'sum'
    # })).shape[0]
    
    


def Q3(vdo_df):
    """
    3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    """
    comment_is_greater_than_10000 = lambda row : row['comment_count'] > 10000
    trending_date_is_18_22_01 = lambda row : row['trending_date'] == '18.22.01'
    return select_where (comment_is_greater_than_10000) (
        select_where (trending_date_is_18_22_01) (
            vdo_df.drop_duplicates()).groupby('title').agg({ 'comment_count':'sum', })).shape[0]



def Q4(vdo_df):
    """
    4. Which trending date that has the minimum average number of comments per VDO?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    grouped = vdo_df.groupby('trending_date').agg({
        'trending_date':'first',
        'comment_count':'mean',
    })
    return  grouped.sort_values(by='comment_count')[:1].trending_date.iloc[0]



def Q5(vdo_df):
    """
    5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
        - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - You must load the additional data from 'GB_category_id.json' into memory before executing any operations.
        - To access 'GB_category_id.json', use the path '/data/GB_category_id.json'.
    """
    with open(json_path,'r') as file:
        content = file.read()
    json_data = json.loads(content)
    category = {item['id'] : item['snippet']['title'] for item in json_data['items']}
    
    flat = lambda xs : sum(xs,[])
    category_id_of = lambda category : lambda value : next(iter(flat([[key] if category[key] == value else [] for key in category.keys()])), None)
    int(category_id_of(category)('Sports'))
    
    sports_id = int(category_id_of(category)('Sports'))
    comedy_id = int(category_id_of(category)('Comedy') )
    
    sports = vdo_df.query('category_id == @sports_id').groupby('trending_date').agg({
        'views':'sum',
    })
    comedy = vdo_df.query('category_id == @comedy_id').groupby('trending_date').agg({
        'views':'sum',
    })
    
    result = sports.groupby('trending_date').agg({
        'views':'sum',
    }) > comedy.groupby('trending_date').agg({
        'views':'sum',
    })
    return int(result.query('views').count().iloc[0])
