import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pprint import pprint

import warnings
warnings.simplefilter("ignore", UserWarning)

DATA_DIR = "./data"

train_data = pd.read_csv(os.path.join(DATA_DIR, "training_set.csv"))
training_meta = pd.read_csv(os.path.join(DATA_DIR, "training_set_metadata.csv"))

def preprocess(flux_data, meta_data):
    meta_data.set_index("object_id", inplace=True)
    flux_data.set_index(["object_id", "passband", "mjd"], inplace=True)
    flux_data = flux_data.unstack("passband")

    agg_dict =      {col: ["mean", "max" , "median", "std", "min"] for col in flux_data.columns[flux_data.columns.get_level_values(0) == "flux"]}
    agg_dict.update({col: ["mean", "max" , "median", "std", "min"] for col in flux_data.columns[flux_data.columns.get_level_values(0) == "flux_err"]})
    agg_dict.update({col: ["mean", "max" , "median", "std", "min"] for col in flux_data.columns[flux_data.columns.get_level_values(0) == "detected"]})

    flux_data = flux_data.groupby("object_id").agg(agg_dict)
    merged = flux_data.merge(meta_data, how="left", left_index=True, right_index=True)
    merged.fillna(0, inplace=True)

    return merged

merged = preprocess(train_data, training_meta)
x_train, x_test, y_train, y_test = train_test_split(merged.loc[:, merged.columns != "target"], merged["target"], test_size=0.33)

clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)

# Build submission
from reader import get_objects_by_id
from tqdm import tqdm

predictions = []
datas = []

dfs = []
metas = []
ids = []

with tqdm(total=3492890, smoothing=0) as pbar:
    for (object_id, df), (object_id_meta, df_meta) in zip(get_objects_by_id(os.path.join(DATA_DIR, "test_set.csv"), chunksize=100000), 
                                                          get_objects_by_id(os.path.join(DATA_DIR, "test_set_metadata.csv"), chunksize=100000)):
        if len(dfs) >= 20000:
            test_data = preprocess(pd.concat(dfs), pd.concat(metas))
            datas.append(test_data)

            pred = clf.predict_proba(test_data)
            pred = np.concatenate((pred, np.zeros((len(dfs), 1))), axis=1)
            pred = np.concatenate((np.array(ids).reshape(-1, 1), pred), axis=1)

            predictions.append(pred)

            dfs = []
            metas = []
            ids = []
        
        ids.append(object_id)
        dfs.append(df)
        metas.append(df_meta)
        pbar.update(1)
        
    
    test_data = preprocess(pd.concat(dfs), pd.concat(metas))
    datas.append(test_data)

    pred = clf.predict_proba(test_data)
    pred = np.concatenate((pred, np.zeros((len(dfs), 1))), axis=1)
    pred = np.concatenate((np.array(ids).reshape(-1, 1), pred), axis=1)

    predictions.append(pred)

submission = pd.DataFrame(np.concatenate(predictions, axis=0))
submission.columns = ["object_id", "class_6","class_15","class_16","class_42","class_52","class_53","class_62",
                      "class_64","class_65","class_67","class_88","class_90","class_92","class_95","class_99"]
submission["object_id"] = pd.to_numeric(submission["object_id"], downcast="integer")
submission.set_index("object_id", inplace=True)
submission.to_csv("submission.csv")
pd.concat(datas).to_csv("aggregated_test_data.csv")
