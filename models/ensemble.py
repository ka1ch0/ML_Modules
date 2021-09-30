from typing import List, Optional

from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


class Stacking:
    def __init__(self, base_models: List[List], meta_model: "meta model of sklearn API or Tensorflow etc.",
                 task: str, n_classies: Optional[int] = None):
        """
        Constructor

        Parameters
        -----------
        base_models : Dict of Sklearn APIs or Tensroflow models.
            Base models of stacking.
        meta_model : Sklearn API or Tensorflow model
            Meta model of stacking.
        task : str
            Kind of task you wanto to solve. 'regression' or 'classification'.
        n_classies : int
            Number of class. You must set this if task is 'classification'.

        Returns
        -----------
        None
        """

        self.meta_model = meta_model
        self.base_models = dict()
        self.n_base = 0
        self.task = task

        if self.task not in ["regression", "classification"]:
            raise Exception("Please set task is regression or classification.")

        self.n_classies = n_classies

        if self.n_classies is None and self.task == "classification":
            raise Exception("If you set task is classification, please set n_classies(int).")

        for base_model in base_models:
            self.n_base += 1
            self.base_models[f"models_{self.n_base}"] = base_model

    def train(self, X: np.ndarray, y: np.ndarray, cv: int = 8,
              batch_size: int = 16, epochs: int = 100):
        """
        Training base models and meta model. Trained models is saved as class variables.
        Intermidiate features is preprocessed by StandardScaler when training model and making predict.
        To change how to preprocess is easy, so please change if you need.

        Parameters
        -----------
        X : np.ndarray
            X of training set.
        y : np.ndarray
            Target of training set.
        cv : int
            Number of fold for cross validation.
        batch_size : int
            Batch size for training neural network. You must set this if you use newral network.
        epochs : int
            Number of epochs for training neural network. You must set this if you use newral network.

        Returns
        ----------
        None
        """

        features = {}
        model_no = 0
        features[f"feature_{model_no}"] = X  # dict for saving intermidiate features.

        # training base models
        print("Start training base models:")
        for models in tqdm(self.base_models.values()):
            model_no += 1
            if self.task == "regression":
                features[f"feature_{model_no}"] = np.empty((X.shape[1],
                                                            len(models)))
            else:
                features[f"feature_{model_no}"] = np.empty((X.shape[1],
                                                            len(models)*self.n_classies))
            kf = KFold(n_splits=cv, random_state=6174, shuffle=True)

            for train_idx, val_idx in kf.split(features[f"feature_{model_no-1}"]):
                train_X, val_X = \
                    features[f"feature_{model_no-1}"][train_idx], features[f"feature_{model_no-1}"][val_idx]
                train_y, val_y = y[train_idx], y[val_idx]

                if model_no > 1:
                    sc = StandardScaler()
                    train_X = sc.fit_transform(train_X)
                    val_X = sc.transform(val_X)

                for i, model in enumerate(models):
                    if "sklearn" in str(type(model)):
                        model.fit(train_X, train_y)
                        if self.task == "regression":
                            pred = model.predict(val_X)
                            features[f"feature_{model_no}"][val_idx, i] = pred.reshape(len(val_idx))
                        else:
                            pred = model.predict_proba(val_X)
                            features[f"feature_{model_no}"][val_idx, i:i+self.n_classies] = pred
                    elif "tensorflow" in str(type(model)):
                        ES = EarlyStopping(monitor="val_loss", patience=5)
                        model.fit(x=train_X, y=train_y, epochs=epochs,
                                  batch_size=batch_size, validation_data=(val_X, val_y),
                                  callbacks=[ES])
                        if self.task == "regression":
                            pred = model.predict(val_X)
                            features[f"feature_{model_no}"][val_idx, i] = pred.reshape(len(val_idx))
                        else:
                            pred = model.predict(val_X)
                            features[f"feature_{model_no}"][val_idx, i:i+self.n_classies] = pred

        # training meta model
        print("Start training meta model:")
        kf = KFold(n_splits=cv, random_state=6174, shuffle=True)

        for train_idx, val_idx in tqdm(kf.split(features[f"feature_{self.n_base}"])):
            train_X, val_X = \
                features[f"feature_{self.n_base}"][train_idx], features[f"feature_{self.n_base}"][val_idx]
            train_y, val_y = y[train_idx], y[val_idx]
            sc = StandardScaler()
            train_X = sc.fit_transform(train_X)
            val_X = sc.transform(val_X)

            if "sklearn" in str(type(self.meta_model)):
                self.meta_model.fit(train_X, train_y)
                if self.task == "regression":
                    pred = self.meta_model.predict(val_X)
                else:
                    pred = self.meta_model.predict_proba(val_X)
            elif "tensorflow" in str(type(self.meta_model)):
                ES = EarlyStopping(monitor="val_loss", patience=5)
                self.meta_model.fit(x=train_X, y=train_y, epochs=epochs,
                                    batch_size=batch_size, validation_data=(val_X, val_y),
                                    callbacks=[ES])
                if self.task == "regression":
                    pred = self.meta_model.predict(val_X)
                else:
                    pred = self.meta_model.predict(val_X)

        print("End training all models")

    def predict(self, X: np.ndarray, cv: int = 8):
        """
        Training base models and meta model. Trained models is saved as class variables.
        Intermidiate features is preprocessed by StandardScaler when training model and making predict.
        To change how to preprocess is easy, so please change if you need.

        Parameters
        -----------
        X : np.ndarray
            X of test set.
        cv : int
            Number of fold for cross validation.

        Returns
        ----------
        pred : np.ndarray
            Prediction using trained models.
        """

        features = {}
        model_no = 0
        features[f"feature_{model_no}"] = X  # dict for saving intermidiate features.

        # making intermidiate features
        print("Start making intermidiate features:")
        for models in tqdm(self.base_models.values()):
            model_no += 1
            if self.task == "regression":
                features[f"feature_{model_no}"] = np.empty((X.shape[1],
                                                            len(models)))
            else:
                features[f"feature_{model_no}"] = np.empty((X.shape[1],
                                                            len(models)*self.n_classies))
            kf = KFold(n_splits=cv, random_state=6174, shuffle=True)

            for train_idx, val_idx in kf.split(features[f"feature_{model_no-1}"]):
                train_X, val_X = \
                    features[f"feature_{model_no-1}"][train_idx], features[f"feature_{model_no-1}"][val_idx]

                if model_no > 1:
                    sc = StandardScaler()
                    train_X = sc.fit_transform(train_X)
                    val_X = sc.transform(val_X)

                for i, model in enumerate(models):
                    if "sklearn" in str(type(model)):
                        if self.task == "regression":
                            pred = model.predict(val_X)
                            features[f"feature_{model_no}"][val_idx, i] = pred.reshape(len(val_idx))
                        else:
                            pred = model.predict_proba(val_X)
                            features[f"feature_{model_no}"][val_idx, i:i+self.n_classies] = pred
                    elif "tensorflow" in str(type(model)):
                        if self.task == "regression":
                            pred = model.predict(val_X)
                            features[f"feature_{model_no}"][val_idx, i] = pred.reshape(len(val_idx))
                        else:
                            pred = model.predict(val_X)
                            features[f"feature_{model_no}"][val_idx, i:i+self.n_classies] = pred

        # making predict using meta model
        print("Start making predict")
        print("-" * 50)
        if "sklearn" in str(type(self.meta_model)):
            if self.task == "regression":
                pred = self.meta_model.predict(features[f"feature_{self.n_base}"])
            else:
                pred = self.meta_model.predict_proba(features[f"feature_{self.n_base}"])

        elif "tensorflow" in str(type(self.meta_model)):
            pred = self.meta_model.predict(features[f"feature_{self.n_base}"])

        print("End making predict")
        return pred