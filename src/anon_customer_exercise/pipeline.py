from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


@dataclass
class PipelineFactory:
    X: pd.DataFrame | pl.DataFrame
    y: pd.Series | pl.Series

    regression_model: RegressorMixin | None = None
    classification_model: ClassifierMixin | None = None
    scaler: Literal['StandardScaler', 'MinMaxScaler'] | None = None
    categorical_cols: list[str] | None = None
    scale_numeric_cols: list[str] | None = None
    impute_mean_cols: list[str] | None = None
    impute_median_cols: list[str] | None = None
    one_hot_encode_cols: list[str] | None = None
    label_encode_cols: list[str] | None = None
    test_size: float = 0.2
    random_state: int = 42

    model_pipeline: Pipeline = field(init=False)

    def __post_init__(self):
        self._pandas_to_polars()
        self._build_pipeline()
        self._split()
        self._fit()
        self._predict()

    def _pandas_to_polars(self) -> None:
        if isinstance(self.X, pd.DataFrame):
            self.X = pl.from_pandas(self.X)
        if isinstance(self.y, pd.Series):
            self.y = pl.from_pandas(self.y)
    def _set_transformers(self) -> None:
        self.transformers: list[tuple[str, TransformerMixin, list[str]]] = []
        if self.impute_mean_cols:
            self.transformers.append(
                ('impute_mean',
                SimpleImputer(strategy='mean'),
                self.impute_mean_cols)
            )
        if self.impute_median_cols:
            self.transformers.append(
                ('impute_median',
                SimpleImputer(strategy='median'),
                self.impute_median_cols)
            )
        if self.one_hot_encode_cols:
            self.transformers.append(
                ('one_hot',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                self.one_hot_encode_cols)
            )
        if self.scale_numeric_cols:
            scaler: TransformerMixin
            match self.scaler:
                case 'StandardScaler':
                    scaler = StandardScaler()
                case 'MinMaxScaler':
                    scaler = MinMaxScaler()
                case _:
                    scaler = StandardScaler()
            self.transformers.append(
                ('scale_numeric',
                scaler,
                self.scale_numeric_cols)
            )
    def _build_pipeline(self) -> None:
        self._set_transformers()
        preprocessor = ColumnTransformer(self.transformers)
        self.model = self.classification_model or self.regression_model
        self.model_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', self.model)
        ])
    def _split(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y,
        test_size=self.test_size,
        random_state=self.random_state
            )

    def _fit(self) -> None:
        self.model_pipeline.fit(self.X_train, self.y_train)

    def _predict(self) -> None:
        self.y_pred = self.model_pipeline.predict(self.X_test)

    @property
    def score(self) -> float | None:
        try:
            if isinstance(self.model, RegressorMixin):
                r2_result = self.model_pipeline.score(self.X_test, self.y_test)
                print(f'R2: {r2_result}')
                return float(r2_result)
            accuracy_result = self.model_pipeline.score(self.X_test, self.y_test)
            print(f'Model accuracy: {accuracy_result}')
            return float(accuracy_result)

        except TypeError as e:
            raise TypeError(f'Error producing a score {e}')

    @property
    def classifier_report(self) ->  None:
        try:
            if isinstance(self.model, ClassifierMixin):
                classification_report(self.y_test, self.y_pred)
        except TypeError as e:
            raise TypeError(f'Classifier reports only available for classification models {e}')