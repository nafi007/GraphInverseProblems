import json
import urllib
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch


class ChickenpoxDatasetLoader(object):
    """A dataset of county level chicken pox cases in Hungary between 2004
    and 2014. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are counties and
    edges are neighbourhoods. Vertex features are lagged weekly counts of the
    chickenpox cases (we included 4 lags). The target is the weekly number of
    cases for the upcoming week (signed integers). Our dataset consist of more
    than 500 snapshots (weeks).

    CUSTOM NOTE: only tested to work for lag=1, which is what you should be using.
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        ###
        data = np.asarray(self.features).squeeze() # has shape [521,20] ...corresponding to 521 weeks for 20 counties in Hungary
        data_3D = np.zeros((data.shape[0], data.shape[1], 2))
        data_3D[:, :, 0] = data # actual data
        # Fill the second channel with incremental values for each row
        for i in range(data_3D.shape[0]):
            data_3D[i, :, 1] = i + 1
        self.features = data_3D

        # let's subtract mean and divide by std
        

        # row 1 (index 0) in data_3D[:,:,1] will now have all 1.0s , row 2 all 2.0s, and so on. Row 521 will have all 521.0s 

        # nfreqs = 10 #data.shape[1] is 20 #this could be arbitrary, but I'm choosing 10 here
        # freqs = np.arange(1, nfreqs + 1).reshape(1, 1, nfreqs) # shape [1,1,20] , entries are 1,2,3...20
        # time_ind_3D = data_3D[:,:,1]  #only time 
        # time_ind_3D = torch.from_numpy(time_ind_3D)
        # test2 = test2.unsqueeze(-1) # shape [521, 20, 1]
        # test3 = test2 * freqs  # shape [521, 20, 20] , each row has time encoding of 1,2...
        # scale_factor = 0.01
        # # test3 = test3 * scale_factor 
        # cosTime = torch.from_numpy(np.cos(2 * np.pi * scale_factor * time))
        # sinTime = torch.from_numpy(np.sin(2 * np.pi * time)).permute(0, 2, 1)
    

        ###
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]



    def get_dataset(self, lags: int = 1) -> StaticGraphTemporalSignal:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset