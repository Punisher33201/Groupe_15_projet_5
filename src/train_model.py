import sys 
sys.path.append('.')

from src.data_processing import load_data

df = load_data()
print(df)