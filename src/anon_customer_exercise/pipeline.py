from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass
class PipelineFactory:
    X: pd.DataFrame 
    y: pd.Series

    model: RegressorMixin | ClassifierMixin | None = None
    scaler: Literal["StandardScaler", "MinMaxScaler"] = "StandardScaler"
    infer_cols: bool = True

    numeric_cols: list[str] | None = None
    numeric_impute_strategy: Literal['median', 'mean'] | None = 'mean'
    categorical_cols: list[str] | None = None

    test_size: float = 0.2
    random_state: int = 42

    model_pipeline: Pipeline = field(init=False)

    def __post_init__(self):
        self._convert_polars()
        self._infer_columns()
        self._split()
        self._build_pipeline()
        self._fit()
        self._predict()

    def _convert_polars(self) -> None:
        if isinstance(self.X, pl.DataFrame):
            self.X = self.X.to_pandas()

        if isinstance(self.y, pl.Series):
            self.y = self.y.to_pandas()

    def _infer_columns(self) -> None:
        if not self.infer_cols:
            return
        if self.numeric_cols is None:
            self.numeric_cols = self.X.select_dtypes(include=["int", "float"]).columns.to_list()
        if self.categorical_cols is None:
            self.categorical_cols = self.X.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
        
    def _build_pipeline(self) -> None:
        scaler = StandardScaler() if self.scaler == "StandardScaler" else MinMaxScaler()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)),
            ("scaler", scaler)
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_cols),
                ("cat", categorical_pipeline, self.categorical_cols)
            ],
            remainder="drop"
        )

        self.model_pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", self.model)
        ])

    def _split(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def _fit(self) -> None:
        self.model_pipeline.fit(self.X_train, self.y_train)
    def _predict(self) -> None:
        self.y_pred = self.model_pipeline.predict(self.X_test)
    @property
    def score(self) -> float:
        return float(self.model_pipeline.score(self.X_test, self.y_test))

    @property
    def classifier_report(self) -> str | None:
        if isinstance(self.model, ClassifierMixin):
            return f'''
            {self.model}
            ======================================================
            {classification_report(self.y_test, self.y_pred)}
            '''
        return None

    @property
    def feature_importances(self) -> pd.DataFrame | None:
        model = self.model_pipeline.named_steps['model']
        cols = self.model_pipeline.named_steps['preprocess'].get_feature_names_out()
        
        if hasattr(model, "coef_"):
            values = np.transpose(model.coef_)
            name = "coefficient"
        elif hasattr(model, "feature_importances_"):
            values = model.feature_importances_.reshape(-1, 1)
            name = "importance"
        else:
            return None

        return pd.DataFrame({ "feature": cols, name: values.flatten() })