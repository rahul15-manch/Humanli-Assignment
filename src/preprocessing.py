from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numerical_cols = [
    "amt",
    "city_pop",
    "age",
    "customer_merchant_distance_km"
]

categorical_cols = [
    "merchant",
    "category",
    "gender",
    "state",
    "job"
]

improved_numerical_cols = [
    "amt",
    "city_pop",
    "age",
    "customer_merchant_distance_km",
    "hour",
    "day",
    "month",
    "day_of_week"
]


def create_preprocessor():
    return ColumnTransformer([
        ("numerical", StandardScaler(), numerical_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])


def create_improved_preprocessor():
    return ColumnTransformer([
        ("numerical", StandardScaler(), improved_numerical_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
