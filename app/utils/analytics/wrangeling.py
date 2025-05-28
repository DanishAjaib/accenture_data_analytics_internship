import pandas as pd
from typing import List

def merge_proportion_dfs(proportion_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    modified_dfs = []

    for df in proportion_dfs:
        category_name = df.columns[0]
        df['category'] = category_name
        df.rename(columns={category_name: 'subcategory'}, inplace=True)
        modified_dfs.append(df)
    
    combined_df = pd.concat(modified_dfs, ignore_index=True)
    combined_df = combined_df[['category', 'subcategory', 'count', 'percentage']]
    
    return combined_df


