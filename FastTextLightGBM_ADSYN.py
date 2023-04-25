import io
import os
import gc
import re
import pickle
import random
import termcolor
import warnings
import shutil
from collections import Counter
from functools import partial
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  accuracy_score

import lightgbm as lgbm
from joblib import dump, load

import fasttext
import texthero
from nltk.corpus import stopwords
from imblearn.under_sampling import RandomUnderSampler

stopwords_list = stopwords.words('english') + stopwords.words('french')


def save_pkl(dir, name, obj):
    dir.mkdir(exist_ok=True)
    with open(dir / name, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(dir, name):
    with open(dir / name, 'rb') as f:
        return pickle.load(f)

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

DEBUG = True
SAVE_PATH = None
SEED = 42
NOW = datetime.strftime(datetime.now() , "%m%d")

SAVE_PATH = Path(f'../input/')
SAVE_PATH.mkdir(exist_ok=True)

NUM_WORKERS = os.cpu_count()

print("DEBUG:\t", DEBUG)
print("SAVE_PATH:\t", SAVE_PATH)
print("NUM_WORKERS:\t", NUM_WORKERS)

    
set_seed(SEED)


train_df = pd.read_csv("./input/Train.csv")
test_df = pd.read_csv("./input/Test.csv")

LABEL2ID = {label:i for i, label in enumerate(train_df['label'].unique())}
ID2LABEL = {v:k for k, v in LABEL2ID.items()}

train_df['label_ids'] = train_df['label'].map(LABEL2ID)

train_targets = train_df['label_ids'].values
all_texts = pd.concat([train_df['text'].str.lower(), test_df['text'].str.lower()])
all_texts = texthero.remove_stopwords(all_texts, stopwords_list)
all_texts = texthero.remove_whitespace(all_texts)

with open("../input/data.txt", "w") as f:
    for line in all_texts:
        f.write(line+"\n")
        
#%%time
fattext_model = fasttext.train_unsupervised("../input/data.txt", model='skipgram', dim=300, wordNgrams=2, epoch=10)

all_features = [fattext_model.get_sentence_vector(text) for text in tqdm(all_texts)]
all_features = np.vstack(all_features)
all_features = np.vstack(all_features)  


train_features = all_features[:len(train_df)]
test_features = all_features[len(train_df):]


# ADSYN version
from imblearn.over_sampling import ADASYN

# Define the ADSYN sampler
adasyn = ADASYN(random_state=42)
# Generate synthetic samples using ADSYN
train_features_res, train_targets_res = adasyn.fit_resample(train_features, train_targets)



from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
import numpy as np

param_distributions = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [16, 32, 64],
    'max_depth': [6, 8, 12],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.01, 0.1, 1.0],
    'reg_lambda': [0.01, 0.1, 1.0],
}

model = LGBMClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1_micro',
    verbose=3,
    random_state=42,
)

random_search.fit(train_features_res, train_targets_res)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}") 

#Best parameters: {'subsample': 0.6, 'reg_lambda': 1.0, 'reg_alpha': 0.1, 'num_leaves': 64, 'n_estimators': 1000, 'min_child_weight': 10, 'max_depth': 8, 'learning_rate': 0.1, 'colsample_bytree': 1.0}
#Best score: 0.8640     
    
