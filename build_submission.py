import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from pprint import pprint

import model

import warnings
warnings.simplefilter("ignore", UserWarning)

DATA_DIR = "./data"
CHUNK = 1000000
BATCH = 20000

train_data = pd.read_csv(os.path.join(DATA_DIR, "training_set.csv"))
training_meta = pd.read_csv(os.path.join(DATA_DIR, "training_set_metadata.csv"))

# clf = model.MLP(hidden_layer_sizes=(100, 100))
clf = model.RandomForest(n_estimators=1000, class_weight="balanced", max_depth=3)
clf.fit(train_data, training_meta)

# Build submission
from reader import get_objects_by_id
from tqdm import tqdm


test_meta_data = pd.read_csv(os.path.join(DATA_DIR, "test_set_metadata.csv"))

with tqdm(total=3492890, smoothing=0) as pbar:

    predictions = []
    for flux_df in pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"), chunksize=CHUNK):
        ids = flux_df["object_id"].unique()
        num_objects = len(ids)
        pred = clf.raw_pred(flux_df, test_meta_data)
        pred = np.concatenate((pred, np.zeros((num_objects, 1))), axis=1)
        pred = np.concatenate((np.array(ids).reshape(-1, 1), pred), axis=1)
        predictions.append(pred)
        pbar.update(num_objects)

submission = pd.DataFrame(np.concatenate(predictions, axis=0))
submission.columns = ["object_id", "class_6","class_15","class_16","class_42","class_52","class_53","class_62",
                      "class_64","class_65","class_67","class_88","class_90","class_92","class_95","class_99"]
submission["object_id"] = pd.to_numeric(submission["object_id"], downcast="integer")
submission = submission.groupby("object_id").mean()
submission.set_index("object_id", inplace=True)
submission.to_csv("submission.csv")
