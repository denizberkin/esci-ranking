import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from utils.logger import log_time
from nltk.util import ngrams



@log_time
def brand_query_alignment(df):
    """ float scoring between 0-1 """
    def calculate_alignment(row):
        query = row['query'].lower()
        brand = row['product_brand'].lower() if not pd.isna(row['product_brand']) else ""
        
        if not brand or len(brand) < 2:  # skip for short samples
            return 0.0
            
        # check exact brand match
        if brand in query:
            # Longer brands in query are more significant
            return min(1.0, len(brand) / (len(query) + 0.1))
        
        # check partial match (brand split into words)
        brand_words = brand.split()
        for word in brand_words:
            if len(word) > 2 and word in query:  # consider words > 2 chars
                return 0.5 * (len(word) / (len(query) + 0.1))
        return 0.0
    
    df['brand_query_alignment'] = df.apply(calculate_alignment, axis=1)
    return df, ['brand_query_alignment']


@log_time
def feature_interactions(df):
    # combine transformer embedding with iou which performs well?
    if 'st_cosine_sim' in df.columns and 'token_overlap' in df.columns:
        df['st_token_interaction'] = df['st_cosine_sim'] * df['token_overlap']
    
    # combine substring ratio with it as well, hope for the same effect
    if 'st_cosine_sim' in df.columns and 'longest_common_substring_ratio' in df.columns:
        df['st_substring_interaction'] = df['st_cosine_sim'] * df['longest_common_substring_ratio']
    
    # non linear version
    if 'st_cosine_sim' in df.columns:
        df['st_cosine_sim_squared'] = np.square(df['st_cosine_sim'])
    
    return df, ['st_token_interaction', 'st_substring_interaction', 'st_cosine_sim_squared']


"""
@log_time
def extract_attribute_match(df):
    # common attributes as I did not have much time extracting from df, 
    # considering I have started adding these features the last day :')
    colors = set(['red', 'blue', 'green', 'black', 'white', 'yellow', 'purple', 
                 'pink', 'brown', 'orange', 'gray', 'grey', 'silver', 'gold'])
    
    sizes = set(['small', 'medium', 'large', 'xl', 'xxl', 'xs', 's', 'm', 'l', 'extra'])
    
    materials = set(['cotton', 'leather', 'wool', 'polyester', 'silk', 'linen', 
                    'plastic', 'metal', 'wood', 'glass', 'ceramic', 'rubber'])
    
    def calculate_attribute_match(row):
        query = set(row['query'].lower().split())
        title = set(row['product_title'].lower().split())
        description = set(row['product_description'].lower().split()) if not pd.isna(row['product_description']) else set()
        
        # Check for color matches
        color_in_query = query.intersection(colors)
        color_in_product = title.union(description).intersection(colors)
        color_match = 1.0 if color_in_query and color_in_query == color_in_product else 0.0
        
        # Check for size matches
        size_in_query = query.intersection(sizes)
        size_in_product = title.union(description).intersection(sizes)
        size_match = 1.0 if size_in_query and size_in_query == size_in_product else 0.0
        
        # Check for material matches
        material_in_query = query.intersection(materials)
        material_in_product = title.union(description).intersection(materials)
        material_match = 1.0 if material_in_query and material_in_query == material_in_product else 0.0
        
        # Overall attribute match score
        total_attributes = len(color_in_query) + len(size_in_query) + len(material_in_query)
        if total_attributes == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        attribute_score = (color_match * len(color_in_query) + 
                          size_match * len(size_in_query) + 
                          material_match * len(material_in_query)) / total_attributes
        
        return color_match, size_match, material_match, attribute_score
    
    attrs = df.apply(calculate_attribute_match, axis=1, result_type='expand')
    df['color_match'] = attrs[0]
    df['size_match'] = attrs[1]
    df['material_match'] = attrs[2]
    df['attribute_match_score'] = attrs[3]
    
    return df, ['color_match', 'size_match', 'material_match', 'attribute_match_score']
    """


def enhanced_feature_extraction(df: pd.DataFrame
                                ) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = []
    
    df, brand_features = brand_query_alignment(df)
    feature_columns.extend(brand_features)
    
    df, interaction_features = feature_interactions(df)
    feature_columns.extend(interaction_features)
    
    # df, attribute_features = extract_attribute_match(df)
    # feature_columns.extend(attribute_features)
    
    return df, feature_columns