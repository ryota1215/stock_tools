import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Union
import matplotlib.pyplot as plt


def ransac_plot(model, x, y, pred_y):
    """
    ransacの異常値除去を可視化
    """
    inlier = model.inlier_mask_
    outlier = np.logical_not(inlier)

    plt.scatter(x[inlier], y[inlier], c='blue',
                edgecolor='black', s=30, marker='o', label='正常値'
                )

    plt.scatter(x[outlier], y[outlier], c='green',
                edgecolor='black', s=30, marker='o', label='外れ値'
                )
    plt.plot(x.values,
             pred_y,
             color='red',
             lw=3,
             label="回帰直線",
             )

    plt.legend()
    plt.show()


def base_params(
    model_name: str = "liner"
):
    """
    model_name : linear,ransac,ridge,lasso,lgb
    scaler_name : ss,mm,ssmm,mmss
    return dict_model
    """
    if model_name == "liner":
        params = {}
    elif model_name == "ransac":
        params = {
            "base_estimator": None,
            "min_samples": None,
            "residual_threshold": None,
            "max_trials": 100,
            "max_skips": np.inf,
            "stop_n_inliers": np.inf,
            "stop_score": np.inf,
            "stop_probability": 0.99,
            "loss": "absolute_error",
            "random_state": 1
        }
    elif model_name == "ridge" or model_name == "lasso":
        params = {"alpha": 1.0}
    elif model_name == "lgb":
        params = params = {
            'objective': 'regression',
            'num_leaves': 10,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'metric': 'rmse',
            "learning_rate": 0.01,
            "extra_trees": True,
            "colsample_bytree": 0.1,
            "random_state": 1,
        }
    return params


def creat_lg_model(
        model_name: str = "liner",
        params: dict = {},
        num_round: int = 100,
        early_stopping_rounds: int = 20,
        verbose_eval: bool = False,
        scaler_name: str = None,
        train_x: Union[pd.Series, pd.DataFrame] = None,
        train_y: Union[pd.Series, pd.DataFrame] = None,
        valid_x: Union[pd.Series, pd.DataFrame] = None,
        valid_y: Union[pd.Series, pd.DataFrame] = None,
):
    """
    model_name : linear,ransac,ridge,lasso,lgb
    scaler_name : ss,mm,ssmm,mmss
    return dict_model
    """
    train_x = train_x.copy()
    train_x = train_x.copy()
    if model_name == "lgb":
        valid_x = valid_x.copy()
        valid_y = valid_y.copy()

    dict_model = {}
    model = select_model(model_name, params)
    dict_scaler = scale(model_name, scaler_name, train_x, train_y, valid_x, valid_y)

    if dict_scaler is not None:
        dict_model["scaler_x"] = dict_scaler["scaler_x"]
        dict_model["scaler_y"] = dict_scaler["scaler_y"]
        if model == "lgb":
            df_scale_x = pd.concat((train_x, valid_x))
            try:
                df_scale_x[:] = dict_scaler["scale_x"]
            except ValueError:
                df_scale_x[:] = dict_scaler["scale_x"].flatten()
            train_x = df_scale_x.reindex(train_x.index)
            valid_x = df_scale_x.reindex(valid_x.index)

            df_scale_y = pd.concat((train_y, valid_y))
            try:
                df_scale_y[:] = dict_scaler["scale_y"]
            except ValueError:
                df_scale_y[:] = dict_scaler["scale_y"].flatten()
            train_y = df_scale_y.reindex(train_y.index)
            valid_y = df_scale_x.reindex(valid_y.index)
        else:
            try:
                train_x[:] = dict_scaler["scale_x"]
            except ValueError:
                train_x[:] = dict_scaler["scale_x"].flatten()
            try:
                train_y[:] = dict_scaler["scale_y"]
            except ValueError:
                train_y[:] = dict_scaler["scale_y"].flatten()

    else:
        dict_model["scaler_x"] = None
        dict_model["scaler_y"] = None

    if model == "lgb":
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(valid_x, valid_y)
        model = lgb.train(params, lgb_train,
                          num_boost_round=num_round,
                          valid_names=['train', 'valid'],
                          valid_sets=[lgb_train, lgb_valid],
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval
                          )
    else:
        if len(train_x.shape) == 1:
            train_x = train_x.values.reshape(-1, 1)
        model = model.fit(train_x, train_y)
    dict_model["model_name"] = model_name
    dict_model["model"] = model
    return dict_model


def select_model(model_name: str = "liner", params: dict = {}):
    if model_name == "liner":
        model = LinearRegression(**params)
    elif model_name == "ransac":
        model = RANSACRegressor(**params)
    elif model_name == "ridge":
        model = Ridge(**params)
    elif model_name == "lasso":
        model = Lasso(**params)
    elif model_name == "lgb":
        model = "lgb"
    return model


def scale(model_name, scaler_name, train_x, train_y, valid_x, valid_y):
    if scaler_name is None:
        return None
    elif scaler_name == "ss":
        scaler_model_x = StandardScaler()
        scaler_model_y = StandardScaler()
    elif scaler_name == "mm":
        scaler_model_x = MinMaxScaler()
        scaler_model_y = MinMaxScaler()
    elif scaler_name == "ssmm":
        scaler_model_x1 = StandardScaler()
        scaler_model_x2 = MinMaxScaler()
        scaler_model_y1 = StandardScaler()
        scaler_model_y2 = MinMaxScaler()
    elif scaler_name == "mmss":
        scaler_model_x1 = MinMaxScaler()
        scaler_model_x2 = StandardScaler()
        scaler_model_y1 = MinMaxScaler()
        scaler_model_y2 = StandardScaler()

    if scaler_name == "ss" or scaler_name == "mm":
        if model_name == "lgb":
            x = pd.concat((train_x, valid_x))
            if len(x.shape) == 1:
                x = x.values.reshape(-1, 1)
            scale_x = scaler_model_x.fit_transform(x)
            y = pd.concat((train_y, valid_y)).values.reshape(-1, 1)
            scale_y = scaler_model_y.fit_transform(y)
        else:
            if len(train_x.shape) == 1:
                train_x = train_x.values.reshape(-1, 1)
            scale_x = scaler_model_x.fit_transform(train_x)
            scale_y = scaler_model_y.fit_transform(train_y.values.reshape(-1, 1))
        return {"scaler_x": scaler_model_x, "scaler_y": scaler_model_y, "scale_x": scale_x, "scale_y": scale_y}

    elif scaler_name == "ssmm" or scaler_name == "mmss":
        if model_name == "lgb":
            x = pd.concat((train_x, valid_x))
            if len(x.shape) == 1:
                x = x.values.reshape(-1, 1)
            scale_x = scaler_model_x1.fit_transform(x)
            scale_x = scaler_model_x2.fit_transform(scale_x)

            y = pd.concat((train_y, valid_y)).values.reshape(-1, 1)
            scale_y = scaler_model_y1.fit_transform(y)
            scale_y = scaler_model_y2.fit_transform(scale_y)
        else:
            if len(train_x.shape) == 1:
                train_x = train_x.values.reshape(-1, 1)
            scale_x = scaler_model_x1.fit_transform(train_x)
            scale_x = scaler_model_x2.fit_transform(scale_x)
            scale_y = scaler_model_y1.fit_transform(train_y.values.reshape(-1, 1))
            scale_y = scaler_model_y2.fit_transform(scale_y)
        return {"scaler_x": [scaler_model_x1, scaler_model_x2], "scaler_y": [scaler_model_y1, scaler_model_y2], "scale_x": scale_x, "scale_y": scale_y}
