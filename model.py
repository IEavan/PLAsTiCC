import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import tsfresh
from lightgbm import LGBMClassifier

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

def series_preprocess(flux_df, params=None):
    if params is None:
        params = {
            "flux": {
                "longest_strike_above_mean": None,
                "longest_strike_below_mean": None,
                "mean_change": None,
                "mean_abs_change": None,
                "length": None,
            },
                    
            "flux_passband": {
                "fft_coefficient": [
                        {"coeff": 0, "attr": "abs"}, 
                        {"coeff": 1, "attr": "abs"}
                    ],
                "kurtosis" : None, 
                "skewness" : None,
            },
                    
            "mjd": {
                "maximum": None, 
                "minimum": None,
                "mean_change": None,
                "mean_abs_change": None,
            },
        }
    
    flux_series_features = tsfresh.extract_features(flux_df.reset_index(),
                                                    default_fc_parameters=params["flux"],
                                                    column_id="object_id",
                                                    column_sort="mjd",
                                                    column_value="flux")
    passband_series_features = tsfresh.extract_features(flux_df.reset_index(),
                                                        default_fc_parameters=params["flux_passband"],
                                                        column_id="object_id",
                                                        column_sort="mjd",
                                                        column_value="flux",
                                                        column_kind="passband")
    df_det = flux_df[flux_df['detected']==1].copy()
    agg_df_mjd = tsfresh.extract_features(df_det.reset_index(), 
                                          column_id='object_id', 
                                          column_value='mjd', 
                                          default_fc_parameters=params['mjd'])
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    flux_series_features.index.rename("object_id", inplace=True)
    passband_series_features.index.rename("object_id", inplace=True)
    agg_df_mjd.index.rename("object_id", inplace=True)

    return pd.concat([flux_series_features,
                      passband_series_features,
                      agg_df_mjd], axis=1)


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def raw_pred(self, flux_df, meta_df):
        pass

    def preprocess(self, flux_df, meta_df, includes_target=True):
        return aggs_preprocess(flux_df, meta_df, includes_target=includes_target)

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


class RF_GAL_EXTRA(BaseModel):

    def __init__(self, *args, **kwargs):
        self.gal_clf = RandomForestClassifier(*args, **kwargs)
        self.extra_clf = RandomForestClassifier(*args, **kwargs)

    def fit(self, flux_df, flux_meta):
        x, y = self.preprocess(flux_df, flux_meta)
        x["target"] = y
        xy_gal = x[x["hostgal_photoz"] == 0]
        xy_extra = x[x["hostgal_photoz"] != 0]

        x_gal = xy_gal.loc[:, xy_gal.columns != "target"]
        y_gal = xy_gal["target"]

        x_extra = xy_extra.loc[:, xy_extra.columns != "target"]
        y_extra = xy_extra["target"]

        self.gal_clf.fit(x_gal, y_gal)
        self.extra_clf.fit(x_extra, y_extra)

    def raw_pred(self, flux_df, meta_df):
        processed_data = self.preprocess(flux_df, meta_df, includes_target=False)
        gal_data = processed_data[processed_data["hostgal_photoz"] == 0]
        extra_data = processed_data[processed_data["hostgal_photoz"] != 0]

        if len(gal_data) != 0:
            gal_pred = self.gal_clf.predict_proba(gal_data)
        else:
            gal_pred = np.zeros((0, len(gal_data.columns)))

        if len(extra_data) != 0:
            extra_pred = self.extra_clf.predict_proba(extra_data)
        else:
            extra_pred = np.zeros((0, len(extra_data.columns)))

        all_classes = list(set(self.gal_clf.classes_).union(set(self.extra_clf.classes_)))
        all_classes = sorted(all_classes)

        gal_pred_full = np.zeros((gal_pred.shape[0], len(all_classes)))
        for i, cls in enumerate(self.gal_clf.classes_):
            gal_pred_full[:, all_classes.index(cls)] = gal_pred[:, i]
        extra_pred_full = np.zeros((extra_pred.shape[0], len(all_classes)))
        for i, cls in enumerate(self.extra_clf.classes_):
            extra_pred_full[:, all_classes.index(cls)] = extra_pred[:, i]

        # recombine preds
        gal_pred = np.concatenate([np.array([gal_data.index.values]).T, gal_pred_full], axis=1)
        extra_pred = np.concatenate([np.array([extra_data.index.values]).T, extra_pred_full], axis=1)
        pred = np.concatenate([gal_pred, extra_pred], axis=0)
        pred = pred[pred[:,0].argsort()]
        return pred[:, 1:]


class LGBM(BaseModel):

    def __init__(self):
        self.best_params = {
                'device': 'cpu', 
                'objective': 'multiclass', 
                'num_class': 14, 
                'boosting_type': 'gbdt', 
                'n_jobs': -1, 
                'max_depth': 7, 
                'n_estimators': 500, 
                'subsample_freq': 2, 
                'subsample_for_bin': 5000, 
                'min_data_per_group': 100, 
                'max_cat_to_onehot': 4, 
                'cat_l2': 1.0, 
                'cat_smooth': 59.5, 
                'max_cat_threshold': 32, 
                'metric_freq': 10, 
                'verbosity': -1, 
                'metric': 'multi_logloss', 
                'xgboost_dart_mode': False, 
                'uniform_drop': False, 
                'colsample_bytree': 0.5, 
                'drop_rate': 0.173, 
                'learning_rate': 0.0267, 
                'max_drop': 5, 
                'min_child_samples': 10, 
                'min_child_weight': 100.0, 
                'min_split_gain': 0.1, 
                'num_leaves': 7, 
                'reg_alpha': 0.1, 
                'reg_lambda': 0.00023, 
                'skip_drop': 0.44, 
                'subsample': 0.75
        }
        self.clf = LGBMClassifier(**self.best_params)

    def preprocess(self, flux_df, meta_df, includes_target=True):
        if includes_target:
            aggs, y = aggs_preprocess(flux_df, meta_df, includes_target=includes_target)
        else:
            aggs = aggs_preprocess(flux_df, meta_df, includes_target=includes_target)
        series = series_preprocess(flux_df)
        merged = series.merge(aggs, on="object_id", how="left")

        if includes_target:
            return merged, y
        else:
            return merged

    def fit(self, flux_df, meta_df):
        x, y = self.preprocess(flux_df, meta_df)
        
        class_counts = y.value_counts()
        weights = {i : np.sum(class_counts) / class_counts[i] for i in class_counts.index}
        self.clf.fit(x, y, sample_weight=y.map(weights),
                     early_stopping_rounds=50)

    def raw_pred(self, flux_df, meta_df):
        processed_data = self.preprocess(flux_df, meta_df, includes_target=False)
        return self.clf.predict_proba(processed_data)
