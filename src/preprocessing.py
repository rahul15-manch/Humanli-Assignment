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


def create_preprocessor():
    return ColumnTransformer([
        ("numerical", StandardScaler(), numerical_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
