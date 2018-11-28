import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod

def aggs_preprocess(flux_df, meta_df, includes_target=True):
    if not meta_df.index.name == "object_id":
        meta_df.set_index("object_id", inplace=True)
    flux_df.set_index(["object_id", "passband", "mjd"], inplace=True)
    flux_df = flux_df.unstack("passband")

    agg_dict =      {col: ["mean", "max" , "median", "std", "min"] for col in flux_df.columns[flux_df.columns.get_level_values(0) == "flux"]}
    agg_dict.update({col: ["mean", "max" , "median", "std", "min"] for col in flux_df.columns[flux_df.columns.get_level_values(0) == "flux_err"]})
    agg_dict.update({col: ["mean", "max" , "median", "std", "min"] for col in flux_df.columns[flux_df.columns.get_level_values(0) == "detected"]})

    flux_df = flux_df.groupby("object_id").agg(agg_dict)
    merged = flux_df.merge(meta_df, how="left", left_index=True, right_index=True)
    merged.fillna(0, inplace=True)
    
    if includes_target:
        x = merged.loc[:, merged.columns != "target"]
        y = merged["target"]
        return x, y
    else:
        return merged

class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def raw_pred(self, flux_df, meta_df):
        pass

    @abstractmethod
    def preprocess(self, flux_df, meta_df, includes_target=True):
        pass

    @abstractmethod
    def fit(self, flux_df, meta_df):
        pass

class MLP(BaseModel):

    def __init__(self, *args, **kwargs):
        self.clf = MLPClassifier(*args, **kwargs)
        self.scaler = StandardScaler()

    def preprocess(self, flux_df, meta_df, includes_target=True):
        return aggs_preprocess(flux_df, meta_df, includes_target=includes_target)

    def fit(self, flux_df, meta_df):
        x, y = self.preprocess(flux_df, meta_df)
        x = self.scaler.fit_transform(x)
        self.clf.fit(x, y)

    def raw_pred(self, flux_df, meta_df):
        processed_data = self.preprocess(flux_df, meta_df, includes_target=False)
        processed_data = self.scaler.transform(processed_data)
        return self.clf.predict_proba(processed_data)

class RandomForest(BaseModel):

    def __init__(self, *args, **kwargs):
        self.clf = RandomForestClassifier(*args, **kwargs)

    def preprocess(self, flux_df, meta_df, includes_target=True):
        return aggs_preprocess(flux_df, meta_df, includes_target=includes_target)

    def fit(self, flux_df, meta_df):
        x, y = self.preprocess(flux_df, meta_df)
        self.clf.fit(x,y)

    def raw_pred(self, flux_df, meta_df):
        processed_data = self.preprocess(flux_df, meta_df, includes_target=False)
        return self.clf.predict_proba(processed_data)
