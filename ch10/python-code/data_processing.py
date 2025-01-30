from pathlib import Path
import pandas as pd
from datasets import Dataset

path = Path('./ch10/python-code/us-patent-phrase-to-phrase-matching')

def load_data(file_name):
    return pd.read_csv(path/file_name)

def prepare_input(df):
    df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
    return df

def create_dataset(df):
    return Dataset.from_pandas(df)
