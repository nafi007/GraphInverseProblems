import os
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from six.moves import urllib
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class METRLADatasetLoader(object):
    """A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles
    County in aggregated 5 minute intervals for 4 months between March 2012
    to June 2012.

    For further details on the version of the sensor network and
    discretization see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(METRLADatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._read_web_data()

    def _download_url(self, url, save_path):  # pragma: no cover
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _read_web_data(self):
        url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

        # Check if zip file is in data folder from working directory, otherwise download
        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "METR-LA.zip")
        ):  # pragma: no cover
            if not os.path.exists(self.raw_data_dir):
                os.makedirs(self.raw_data_dir)
            self._download_url(url, os.path.join(self.raw_data_dir, "METR-LA.zip"))

        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "adj_mat.npy")
        ) or not os.path.isfile(
            os.path.join(self.raw_data_dir, "node_values.npy")
        ):  # pragma: no cover
            with zipfile.ZipFile(
                os.path.join(self.raw_data_dir, "METR-LA.zip"), "r"
            ) as zip_fh:
                zip_fh.extractall(self.raw_data_dir)

        A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
        X = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose(
            (1, 2, 0)
        )
        X = X.astype(np.float32)

        nfreqs = 10
        freqs = np.arange(1, nfreqs + 1).reshape(1, 1, nfreqs)
        time = np.expand_dims(X[:, 1, :].copy(), -1) * freqs

        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)
        cosTime = torch.from_numpy(np.cos(2 * np.pi * time)).permute(0, 2, 1)
        sinTime = torch.from_numpy(np.sin(2 * np.pi * time)).permute(0, 2, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

        self.X = torch.cat([self.X[:, 0, :].unsqueeze(1), cosTime, sinTime], dim=1)

        self.means = means
        self.stds = stds

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
