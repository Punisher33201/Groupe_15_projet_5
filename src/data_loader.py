from ucimlrepo import fetch_ucirepo 
import pandas as pd


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    # fetch dataset 
    regensburg_pediatric_appendicitis = fetch_ucirepo(id=938) 
    
    # data (as pandas dataframes) 
    X = regensburg_pediatric_appendicitis.data.features 
    y = regensburg_pediatric_appendicitis.data.targets 

    return X, y

def test_load_data():
    X, y = load_data()
    print(type(X), type(y))

if __name__ == "__main__":
    test_load_data()