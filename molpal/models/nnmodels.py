"""This module contains Model implementations that utilize an NN model as their
underlying model"""
from cgi import test
from functools import partial
import json
from pathlib import Path
from typing import Callable, Iterable, List, NoReturn, Optional, Sequence, Tuple, TypeVar
import numpy as np
from numpy import ndarray
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import ray 
from ray import cluster_resources
from tqdm import tqdm

import torch.nn as nn
import torch 
from torch.utils.data import Dataset, random_split, DataLoader

from molpal.featurizer import Featurizer, feature_matrix
from molpal.models.base import Model

T = TypeVar("T")
T_feat = TypeVar("T_feat")
# Dataset = tf.data.Dataset


class FingerprintDataset(Dataset):
    """ A pytorch dataset containing a list of molecular fingerprints and output values """
    def __init__(self, xs, ys, featurizer: Callable[[T], ndarray]):
        self.X = torch.Tensor(np.array(feature_matrix(xs, featurizer))) # Need to change to featurize() not feature_matrix()
        self.y = torch.Tensor(self._normalize(ys))
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len 
    
    def _normalize(self, ys: Sequence[float]) -> ndarray:
        Y = np.stack(list(ys))
        self.mean = np.nanmean(Y, axis=0)
        self.std = np.nanstd(Y, axis=0)

        return (Y - self.mean) / self.std


def mve_loss(y_true, y_pred):
    if not isinstance(y_pred,torch.Tensor): y_pred = torch.Tensor(y_pred)
    if not isinstance(y_true,torch.Tensor): y_true = torch.Tensor(y_pred)
    mu = y_pred[:,0]
    sp = nn.Softplus()
    var = sp(y_pred[:,1])

    return torch.mean(
        torch.log(torch.tensor(2 * 3.141592)) / 2
        + torch.log(var) / 2
        + torch.square(mu - y_true) / (2 * var)
    )

def make_dataloaders(xs, ys, featurizer, batch_size):
    ''' Make a pytorch dataloader from xs (list of smiles strings) and 
    ys (some outputs). Data is featurized using the provided featurizer '''
    dataset = FingerprintDataset(xs, ys, featurizer)
    lengths = [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))]
    train_data, val_data = random_split(dataset, lengths)
    train_dataloader = DataLoader(
            train_data, 
            batch_size=batch_size)
            # num_workers=int(ray.cluster_resources()['CPU']))
    val_dataloader = DataLoader(
            val_data, 
            batch_size=batch_size)
            # num_workers=int(ray.cluster_resources()['CPU']))
    return train_dataloader, val_dataloader

class NN(LightningModule):
    """A feed-forward neural network model

    Attributes
    ----------
    model : keras.Sequential
        the underlying model on which to train and perform inference with
    optimizer : keras.optimizers.Adam
        the model optimizer
    loss : Callable
        the loss function to use
    input_size : int
        the dimension of the model input
    output_size : int
        the dimension of the model output
    batch_size : int
        the size to batch training into
    uncertainty : Optional[str]
        the uncertainty method this model is using (if at all)
    uncertainty : bool
       Whether the model directly predicts its own uncertainty
    mean : float
        the mean of the unnormalized data
    std : float
        the standard deviation of the unnormalized data

    Parameters
    ----------
    input_size : int
    num_tasks : int
    batch_size : int, default=4096
    layer_sizes : Optional[Sequence[int]], default=None
        the sizes of the hidden layers in the network. If None, default to
        two hidden layers with 100 neurons each.
    dropout : Optional[float], default=None
        if specified, add a dropout hidden layer with the specified dropout
        rate after each hidden layer
    activation : Optional[str], default='relu'
        the name of the activation function to use
    uncertainty : Optional[str], default=None
    """

    def __init__(
        self,
        input_size: int,
        num_tasks: int,
        batch_size: int = 4096,
        layer_sizes: Optional[Sequence[int]] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = "relu",
        uncertainty: Optional[str] = None,
        model_seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size

        self.uncertainty = uncertainty

        layer_sizes = layer_sizes or [100, 100]
        self.model, self.optimizer, self.loss = self.build(
            input_size, num_tasks, layer_sizes, dropout, self.uncertainty, activation
        )

        self.mean = 0
        self.std = 0

        if model_seed: torch.manual_seed(model_seed)

    def build(self, input_size, num_tasks, layer_sizes, dropout, uncertainty, activation):
        """Build the model, optimizer, and loss function"""
        dropout_at_predict = uncertainty == "dropout"
        output_size = 2 * num_tasks if self.uncertainty else num_tasks

        activations = {'relu': nn.ReLU(), 
                'tanh': nn.Tanh(), 
                'sigmoid': nn.Sigmoid(), 
                'leakyrelu': nn.LeakyReLU()
                }

        model = nn.Sequential(nn.Linear(input_size, layer_sizes[0]))

        for i in range(1,len(layer_sizes)):
            model.append(activations[activation])
            if dropout: 
                model.append(nn.Dropout(p=dropout))
            model.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
        
        model.append(activations[activation])
        model.append(nn.Linear(layer_sizes[-1],output_size)) 

        if uncertainty not in {"mve"}:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        elif uncertainty == "mve":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        else:
            raise ValueError(f'Unrecognized uncertainty method: "{uncertainty}"')

        if uncertainty not in {"mve"}:
            loss = nn.MSELoss()
        elif uncertainty == "mve":
            loss = mve_loss
        else:
            raise ValueError(f'Unrecognized uncertainty method: "{uncertainty}"')

        return model, optimizer, loss
    
    def configure_optimizers(self):
        if self.uncertainty not in {"mve"}:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        elif self.uncertainty == "mve":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        else:
            raise ValueError(f'Unrecognized uncertainty method: "{self.uncertainty}"')
        return optimizer
    
    def training_step(self, train_batch: tuple, batch_idx):
        X, y = train_batch
        loss = self.loss(y.float(), self.model(X.float()))
        return loss 
    
    def validation_step(self, batch: tuple, batch_idx):
        X, y = batch
        loss = self.loss(y.float(), self.model(X.float()))
        self.log("val_loss", loss, prog_bar=True)
        return loss 

    def forward(self, x) -> ndarray:
        Y_pred = self.model(x)

        if self.uncertainty == "mve":
            Y_pred[:, 0::2] = Y_pred[:, 0::2] * self.std + self.mean
            Y_pred[:, 1::2] = Y_pred[:, 1::2] * self.std ** 2
        else:
            Y_pred = Y_pred * self.std + self.mean

        return Y_pred

    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = f"{path}/model"
        torch.save(self.model.state_dict(), model_path)
        # self.model.save(model_path, include_optimizer=True)

        state_path = f"{path}/state.json"
        state = {"std": self.std, "mean": self.mean, "model_path": model_path}
        json.dump(state, open(state_path, "w"), indent=4)

        return state_path

    def load(self, path):
        state = json.load(open(path, "r"))

        model_path = state["model_path"]
        self.std = state["std"]
        self.mean = state["mean"]

        if self.uncertainty == "mve":
            custom_objects = {"mve_loss": mve_loss}
        else:
            custom_objects = {}

        self.model.load_state_dict(torch.load(model_path))
        # self.model = keras.models.load_model(model_path, custom_objects=custom_objects)


    def _normalize(self, ys: Sequence[float]) -> ndarray:
        Y = np.stack(list(ys))
        self.mean = np.nanmean(Y, axis=0)
        self.std = np.nanstd(Y, axis=0)

        return (Y - self.mean) / self.std


class NNModel(Model):
    """A simple feed-forward neural network model

    Attributes
    ----------
    model : NN
        the underlying neural net on which to train and perform inference

    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training

    See also
    --------
    NNDropoutModel
    NNEnsembleModel
    NNTwoOutputModel
    """

    def __init__(
        self,
        input_size: int,
        test_batch_size: Optional[int] = 4096,
        dropout: Optional[float] = 0.0,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        self.build_model = partial(
            NN,
            input_size=input_size,
            num_tasks=1,
            batch_size=test_batch_size,
            dropout=dropout,
            model_seed=model_seed,
        )
        self.model = self.build_model()
        self.batch_size = test_batch_size
        super().__init__(test_batch_size=test_batch_size)

    @property
    def provides(self):
        return {"means"}

    @property
    def type_(self):
        return "nn"

    def train(
        self,
        xs: Iterable[T],
        ys: Sequence[Optional[float]],
        *,
        featurizer: Featurizer,
        retrain: bool = False,
        epochs: int = 500,
    ) -> bool:

        if retrain:
            self.model = self.build_model()

        self.train_dataloader, self.val_dataloader = make_dataloaders(xs, ys, featurizer, self.batch_size)

        self.trainer = pl.Trainer(
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                max_epochs=epochs,
                # log_every_n_steps=len(self.train_dataloader),
                )
        self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader) 

        return self.model

    def get_means(self, xs: List) -> ndarray:
        return self.model(xs)[:, 0]

    def get_means_and_vars(self, xs: List) -> NoReturn:
        raise TypeError("NNModel can't predict variances!")

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)



class NNEnsembleModel(Model):
    """A feed-forward neural network ensemble model for estimating mean
    and variance.

    Attributes
    ----------
    models : List[NN]
        the underlying neural nets on which to train and perform inference

    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    ensemble_size : int (Default = 5)
        the number of separate models to train
    bootstrap_ensemble : bool
        NOTE: UNUSED
    """

    def __init__(
        self,
        input_size: int,
        test_batch_size: Optional[int] = 4096,
        dropout: Optional[float] = 0.0,
        ensemble_size: int = 5,
        bootstrap_ensemble: Optional[bool] = False,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 4096
        self.build_model = partial(
            NN,
            input_size=input_size,
            num_tasks=1,
            batch_size=test_batch_size,
            dropout=dropout,
            model_seed=model_seed,
        )

        self.ensemble_size = ensemble_size
        self.models = [self.build_model() for _ in range(self.ensemble_size)]
        self.batch_size = test_batch_size
        self.bootstrap_ensemble = bootstrap_ensemble  # TODO: Actually use this

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def type_(self):
        return "nn"

    @property
    def provides(self):
        return {"means", "vars"}

    def train(
        self,
        xs: Iterable[T],
        ys: Sequence[Optional[float]],
        *,
        featurizer: Featurizer,
        retrain: bool = False,
        epochs: int = 500
    ):
        if retrain:
            self.models = [self.build_model() for _ in range(self.ensemble_size)]
        
        self.train_dataloader, self.val_dataloader = make_dataloaders(xs, ys, featurizer, self.batch_size)

        self.trainers = [ pl.Trainer(
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                max_epochs=epochs,
                # log_every_n_steps=len(self.train_dataloader),
                ) for _ in range(self.ensemble_size)]
        
        for i in tqdm(range(self.ensemble_size),desc='Ensemble Progress'):
            self.trainers[i].fit(self.models[i], self.train_dataloader, self.val_dataloader) 
      
        return self.models

    def get_means(self, xs: Sequence) -> np.ndarray:
        preds = np.zeros((len(xs), len(self.models)))
        for j, model in tqdm(
            enumerate(self.models), leave=False, desc="ensemble prediction", unit="model"
        ):
            preds[:, j] = model.predict(xs)[:, 0]

        return np.mean(preds, axis=1)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros((len(xs), len(self.models)))
        for j, model in tqdm(
            enumerate(self.models), leave=False, desc="ensemble prediction", unit="model"
        ):
            preds[:, j] = model.predict(xs)[:, 0]

        return np.mean(preds, axis=1), np.var(preds, axis=1)

    def save(self, path) -> str:
        for i, model in enumerate(self.models):
            model.save(path, f"model_{i}")

        return path

    def load(self, path):
        for model, model_path in zip(self.models, path.iterdir()):
            model.load(model_path)


class NNTwoOutputModel(Model):
    """Feed forward neural network with two outputs so it learns to predict
    its own uncertainty at the same time

    Attributes
    ----------
    model : NN
        the underlying neural net on which to train and perform inference

    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    """

    def __init__(
        self,
        input_size: int,
        test_batch_size: Optional[int] = 4096,
        dropout: Optional[float] = 0.0,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(
            NN,
            input_size=input_size,
            num_tasks=1,
            batch_size=test_batch_size,
            dropout=dropout,
            uncertainty="mve",
            model_seed=model_seed,
        )
        self.model = self.build_model()

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def type_(self):
        return "nn"

    @property
    def provides(self):
        return {"means", "vars"}

    def train(
        self,
        xs: Iterable[T],
        ys: Sequence[Optional[float]],
        *,
        featurizer: Featurizer,
        retrain: bool = False,
    ) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, featurizer)

    def get_means(self, xs: Sequence) -> np.ndarray:
        preds = self.model.predict(xs)
        return preds[:, 0]

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        preds = self.model.predict(xs)
        return preds[:, 0], self._safe_softplus(preds[:, 1])

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)

    @classmethod
    def _safe_softplus(cls, xs):
        in_range = xs < 100
        return np.log(1 + np.exp(xs * in_range)) * in_range + xs * (1 - in_range)


class NNDropoutModel(Model):
    """Feed forward neural network that uses MC dropout for UQ

    Attributes
    ----------
    model : NN
        the underlying neural net on which to train and perform inference
    dropout_size : int
        the number of forward passes to perform to through the model at inference time

    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    dropout_size : int (Default = 10)
        the number of passes to make through the network during inference
    """

    def __init__(
        self,
        input_size: int,
        test_batch_size: Optional[int] = 4096,
        dropout: Optional[float] = 0.2,
        dropout_size: int = 10,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(
            NN,
            input_size=input_size,
            num_tasks=1,
            batch_size=test_batch_size,
            dropout=dropout,
            uncertainty="dropout",
            model_seed=model_seed,
        )
        self.model = self.build_model()
        self.dropout_size = dropout_size

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def type_(self):
        return "nn"

    @property
    def provides(self):
        return {"means", "vars", "stochastic"}

    def train(
        self,
        xs: Iterable[T],
        ys: Sequence[Optional[float]],
        *,
        featurizer: Featurizer,
        retrain: bool = False,
    ) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, featurizer)

    def get_means(self, xs: Sequence) -> ndarray:
        predss = self._get_predss(xs)
        return np.mean(predss, axis=1)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        predss = self._get_predss(xs)
        return np.mean(predss, axis=1), np.var(predss, axis=1)

    def _get_predss(self, xs: Sequence) -> ndarray:
        """Get the predictions for each dropout pass"""
        predss = np.zeros((len(xs), self.dropout_size))
        for j in tqdm(
            range(self.dropout_size), leave=False, desc="bootstrap prediction", unit="pass"
        ):
            predss[:, j] = self.model.predict(xs)[:, 0]

        return predss

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)
