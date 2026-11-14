import numpy as np
import pandas as pd


def prepare_features(data):
    data = data.copy()

    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
    data["dob"] = pd.to_datetime(data["dob"])

    data["hour"] = data["trans_date_trans_time"].dt.hour
    data["day"] = data["trans_date_trans_time"].dt.day
    data["month"] = data["trans_date_trans_time"].dt.month
    data["day_of_week"] = data["trans_date_trans_time"].dt.dayofweek

    data["age"] = (
        (data["trans_date_trans_time"] - data["dob"]).dt.days / 365.25
    ).astype(int)

    lat1 = np.radians(pd.to_numeric(data["lat"]))
    lon1 = np.radians(pd.to_numeric(data["long"]))
    lat2 = np.radians(pd.to_numeric(data["merch_lat"]))
    lon2 = np.radians(pd.to_numeric(data["merch_long"]))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    a = np.clip(a, 0, 1)

    data["customer_merchant_distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    drop_cols = [
        "Unnamed: 0",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "zip",
        "trans_date_trans_time",
        "dob",
        "trans_num",
        "unix_time",
        "lat",
        "long",
        "merch_lat",
        "merch_long"
    ]

    data.drop(columns=drop_cols, inplace=True, errors="ignore")

    return data
