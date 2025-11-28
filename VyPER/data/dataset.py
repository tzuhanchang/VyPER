import os
import shutil
import yaml
import h5py
import math
import torch

import os.path as osp
import numpy as np
import numpy.lib.recfunctions as rf

from torch import Tensor
from torch_geometric.data import Dataset, Data
from torch_hep.lorentz import MomentumTensor
from itertools import permutations, combinations
from typing import Tuple

from .transform import TransformFeatures


class VyPERDataset(Dataset):
    def __init__(self, root: str, config: str, training: bool=True,
                 cache_dir: str=None, force_reload: bool=False) -> None:
        self.root = root
        self.config = config
        self._train_mode = training

        if cache_dir is None:
            self.cache = osp.join(".cache",osp.splitext(osp.basename(self.root))[0])
        else:
            self.cache = osp.join(cache_dir,osp.splitext(osp.basename(self.root))[0])
        if not osp.exists(self.cache):
            os.makedirs(self.cache)
        else:
            if force_reload:
                shutil.rmtree(self.cache)
                os.makedirs(self.cache)

        config = self._parse_config_file(self.config)
        self.node_input_names = list(config['input']['nodes'].keys())
        self.input_id = config['input']['nodes']

        # Locate 4 vector input features from all node features
        node_4vector_loc = []
        for input in config['input']['node_4vector_definition']['ordered_inputs']:
            loc = 0
            for feat in config['input']['node_features']:
                if input == feat:
                    node_4vector_loc.append(loc)
                loc += 1
        self.node_4vector_loc = torch.LongTensor(node_4vector_loc)
        del node_4vector_loc
        assert self.node_4vector_loc.size(0) == 4

        # Define the function that handles 4 momentum calculation
        self.momentum_func = eval(config['input']['node_4vector_definition']['functional'])

        # Read functions that define edge features
        self.edge_feat_func = [
            eval(f"lambda e1,e2: {func}") for func in config['input']['edge_features']]

        # Read topology definiation
        self.topo_max_num_edges = int(config['target']['topology']['edge'])
        self.topo_max_num_hyperedges = int(config['target']['topology']['hyperedge'])
        self.topo_num_neutrinos = int(config['target']['topology']['neutrinos'])

        # Locate neutrino 4 vector inputs
        self._use_diffusion = True if 'neutrinos' in config['target'].keys() else False
        if self._use_diffusion:
            assert self.topo_num_neutrinos > 0
            self.neutrino_association = torch.tensor(config['target']['neutrinos']['associated_nodes'])
            neutrino_4vector_func = []
            neutrino_4vector_loc = []
            for input in config['target']['neutrinos']['4vector_definition']['ordered_inputs']:
                loc = 0
                func_like = []
                for feat in config['target']['neutrinos']['features']:
                    if feat in input and feat != input:
                        func_like.append(loc)
                    if feat == input:
                        neutrino_4vector_loc.append([loc])
                        neutrino_4vector_func.append(eval(f"lambda {input}: {input}"))
                    loc += 1
                if len(func_like) > 0:
                    neutrino_4vector_loc.append(func_like)
                    neutrino_4vector_func.append(
                        eval(f"lambda {','.join(map(str,np.array(config['target']['neutrinos']['features'])[func_like]))}: {input}"))
            self.neutrino_4vector_loc = neutrino_4vector_loc
            self.neutrino_4vector_func = neutrino_4vector_func
            del neutrino_4vector_loc
            del neutrino_4vector_func
            assert len(self.neutrino_4vector_loc) == len(self.neutrino_4vector_func)
            self.neutrino_momentum_func = eval(config['target']['neutrinos']['4vector_definition']['functional'])
        else:
            assert self.topo_num_neutrinos == 0
            self.neutrino_association = None
            self.neutrino_4vector_loc = None
            self.neutrino_4vector_func = None
            self.neutrino_momentum_func = None

        # Read target edge labels
        self._use_edge = True if 'edge' in config['target'].keys() else False
        self.target_edge_cantor = None
        self.edge_out_channels = None
        if self._use_edge:
            edge_cantor = {}
            for key, value in config['target']['edge'].items():
                edge_cantor.update({key: []})
                for target in value:
                    assert len(target) == 2
                    edge_cantor[key].append(
                        [self._cantor_pairing(*[int(x) for x in target[i].split('-')]) for i in range(len(target))])
                edge_cantor[key] = torch.tensor(edge_cantor[key],dtype=torch.float32).transpose(0,1)
            self.target_edge_cantor = edge_cantor
            self.edge_out_channels = len(self.target_edge_cantor.keys()) + 1
            del edge_cantor
        else:
            assert self.topo_max_num_edges == 0

        # Read target hyperedge labels
        self._use_hyperedge = True if 'hyperedge' in config['target'].keys() else False
        self.target_hyperedge_cantor = None
        self.hyperedge_out_channels = None
        self.hyperedge_exclusion = None
        if self._use_hyperedge:
            assert self.topo_max_num_hyperedges > 0
            hyperedge_cantor = {}
            hyperedge_exclusion = []
            for key, value in config['target']['hyperedge'].items():
                hyperedge_cantor.update({key: []})
                self.hyperedge_order = len(value[0])
                for target in value:
                    assert self.hyperedge_order == len(target)
                    hyperedge_cantor[key].append(
                        [self._cantor_pairing(*[int(x) for x in target[i].split('-')]) for i in range(len(target))])
                    hyperedge_exclusion.append([int(target[i].split('-')[0]) for i in range(len(target))])
                hyperedge_cantor[key] = torch.tensor(hyperedge_cantor[key],dtype=torch.float32).transpose(0,1)
            self.target_hyperedge_cantor = hyperedge_cantor
            self.hyperedge_out_channels = len(self.target_hyperedge_cantor.keys()) + 1
            self.hyperedge_exclusion = torch.tensor(
                list(set(self.input_id.values()).difference(set(np.unique(np.array(hyperedge_exclusion).flatten()))))
            )
            del hyperedge_cantor
            del hyperedge_exclusion
        else:
            assert self.topo_max_num_hyperedges == 0

        # Get input channel size
        self.node_in_channels = len(config['input']['node_features']) + 1
        self.edge_in_channels = len(config['input']['edge_features'])
        self.glob_in_channels = len(config['input']['global_features'])
        self.nu_out_channels  = len(config['target']['neutrinos']['features']) if self._use_diffusion else None

        # Open the HDF5 file for this dataset instance
        self.file = h5py.File(self.root, 'r')

        # Get the number of the entries
        self.size = len(self.file['INPUTS'][self.node_input_names[0]])

        # Assertions
        for feat in self.node_input_names:
            assert config['input']['node_features']==list(self.file['INPUTS'][feat].dtype.names)
        assert config['input']['global_features']==list(self.file['INPUTS/GLOBAL'].dtype.names)

        # Read METADATA
        if 'METADATA' in self.file.keys():
            for key, value in self.file['METADATA'].items():
                setattr(self, f"METADATA_{key}", torch.tensor(np.array(value)))

        # Transformations
        self.node_transform_methods = [
            eval(f"lambda x: {func}")for func in config['input']['node_transforms']]
        self.edge_transform_methods = [
            eval(f"lambda x: {func}")for func in config['input']['edge_transforms']]
        self.glob_transform_methods = [
            eval(f"lambda x: {func}")for func in config['input']['global_transforms']]
        if self._use_diffusion:
            self.nu_transform_methods = [
                eval(f"lambda x: {func}")for func in config['target']['neutrinos']['transforms']]
            self.nu_reverse_transform_methods = [
                eval(f"lambda x: {func}")for func in config['target']['neutrinos']['reverse_transforms']]
        else:
            self.nu_transform_methods = None
            self.nu_reverse_transform_methods = None
        if self._train_mode:
            transform = TransformFeatures(['x', 'edge_attr', 'u', 'neutrino_t'] if self._use_diffusion \
                                     else ['x', 'edge_attr', 'u'],
                                          [self.node_transform_methods,
                                           self.edge_transform_methods,
                                           self.glob_transform_methods,
                                           self.nu_transform_methods] if self._use_diffusion \
                                     else [self.node_transform_methods,
                                           self.edge_transform_methods,
                                           self.glob_transform_methods])
        else:
            transform = TransformFeatures(['x', 'edge_attr', 'u'],
                                          [self.node_transform_methods,
                                           self.edge_transform_methods,
                                           self.glob_transform_methods])

        super().__init__(root, transform=transform, pre_transform=None,
                         pre_filter=None)

    @staticmethod
    def _parse_config_file(filename):
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError:
                raise RuntimeError("Configuration file is broken.")

    @staticmethod
    def _cantor_pairing(src, other):
        r"""Cantor pairing function.

        Note:
            The unique ID is created using the Cantor pairing
            function:
            .. math::
                \pi(k_1,k_2) = \frac{1}{2}(k_1+k_2)(k_1+k_2+1)+k_2
            where k_1 is the `input_id` and k_2 is the `node_id`.
        """
        return (src+other)*(src+other+1)/2 + src

    def build_node_attr(self, INPUTS: h5py._hl.group.Group, index: int) -> Tensor:
        r"""Construct node input tensor from :obj:`INPUTS` HDF5 data group.
        Returning node input tensor.

        Args:
            INPUTS (h5py._hl.group.Group): HDF5 data group for inputs.

        :rtype: :class:`Tensor`
        """
        inputs = []
        for input, id in self.input_id.items():
            values = torch.tensor(rf.structured_to_unstructured(INPUTS[input][index]),
                                  dtype=torch.float32)
            ids = torch.full((values.size(0),1),id,dtype=torch.float32)
            inputs.append(torch.cat([values,ids], dim=1))
        inputs = torch.cat(inputs, dim=0)
        mask = ~torch.any(inputs.isnan(),dim=1)
        return inputs[mask,:]

    def build_edge_attr(self, x: Tensor) -> Tuple[Tensor,Tensor]:
        r"""Construct edge input tensor using node input :obj:`x`.
        Returning edge input tensor and edge index tensor.

        Args:
            x (Tensor): Node input tensor.

        :rtype: :class:`Tuple[Tensor,Tensor]`
        """
        num_nodes = x.size(0)
        edge_index = torch.tensor(list(permutations(range(num_nodes), r=2)),
                                  dtype=torch.int64).permute(dims=(1,0))

        e1 = self.momentum_func(x.index_select(1,self.node_4vector_loc)[edge_index[0]])
        e2 = self.momentum_func(x.index_select(1,self.node_4vector_loc)[edge_index[1]])
        edge_attr = torch.cat([func(e1,e2) for func in self.edge_feat_func],dim=1)
        return edge_index, edge_attr

    def build_glob_attr(self, INPUTS: h5py._hl.group.Group, index: int) -> Tensor:
        r"""Construct global input tensor from :obj:`INPUTS` HDF5 data group.
        Returning global input tensor.

        Args:
            INPUTS (h5py._hl.group.Group): HDF5 data group for inputs.

        :rtype: :class:`Tensor`
        """
        return torch.tensor(rf.structured_to_unstructured(INPUTS['GLOBAL'][index]),
                            dtype=torch.float32)

    def build_hyperedge_index(self, x: Tensor) -> Tensor:
        r"""Construct hyperedge index tensor from :obj:`x` node input tensor.
        Returning hyperedge index tensor.

        Args:
            x (Tensor): Node input tensor.

        :rtype: :class:`Tensor`
        """
        # Excluding unnecessary combinations
        node_loc = torch.all(x[:,-1].unsqueeze(1)!=self.hyperedge_exclusion.unsqueeze(0),
                             dim=1).nonzero().flatten()
        hyperedge_index = torch.tensor(list(combinations(node_loc.tolist(), r=self.hyperedge_order))).transpose(0,1)
        return hyperedge_index

    def get_node_cantor_id(self, LABELS: h5py._hl.group.Group, index: int) -> Tensor:
        r"""Get node Cantor ID using node ID and object truth ID from
        :obj:`LABELS` HDF5 data group.
        Returning node Cantor results.

        Args:
            LABELS (h5py._hl.group.Group): HDF5 data group for targets.

        :rtype: :class:`Tensor`
        """
        inputs = []
        for input, id in self.input_id.items():
            values = torch.tensor(LABELS[input][index],dtype=torch.float32).view(-1,1)
            ids = torch.full((values.size(0),1),id,dtype=torch.float32)
            inputs.append(torch.cat([ids,values], dim=1))
        inputs = torch.cat(inputs, dim=0)
        mask = ~torch.any(inputs.isnan(),dim=1)
        k1, k2 = inputs[mask,:].transpose(0,1)
        return self._cantor_pairing(k1, k2)

    def build_edge_target(self, edge_cantor_id: Tensor,
                          edge_index: Tensor) -> Tensor:
        r"""Construct edge target tensor according to the edge targets
        defined in the configuration file. Searching for edges that have
        matched targeting edge Cantor IDs.
        Returning edge target tensor and edge Cantor.

        Args:
            edge_cantor_id (Tensor): Edge Cantor IDs.
            edge_index (Tensor): Edge index tensor.

        :rtype: :class:`Tensor`
        """
        edge_search = edge_cantor_id.unsqueeze(2)
        edge_attr_t = torch.zeros((edge_index.size(1),self.edge_out_channels),dtype=torch.float32)
        loc = 0
        for key, value in self.target_edge_cantor.items():
            cantor_id = value.unsqueeze(1)
            cantor_id_flip = value.flip(0).unsqueeze(1)
            matches = torch.all(edge_search==cantor_id,dim=0) | torch.all(edge_search==cantor_id_flip, dim=0)
            indices = matches.nonzero()[:,0]
            edge_attr_t[indices,loc] = 1
            loc += 1
        edge_attr_t[~torch.any(edge_attr_t==1,dim=1),self.edge_out_channels-1] = 1
        return edge_attr_t

    def build_neutrino_target(self, LABELS: h5py._hl.group.Group, index: int) -> Tensor:
        nu = torch.tensor(rf.structured_to_unstructured(LABELS['NEUTRINO'][index]),
                          dtype=torch.float32)
        mask = ~torch.any(nu.isnan(),dim=1)
        return nu[mask,:]

    def build_hyperedge_target(self, hyperedge_cantor_id: Tensor,
                               hyperedge_index: Tensor) -> Tensor:
        hyperedge_search = torch.sort(hyperedge_cantor_id,0)[0].unsqueeze(2)
        hyperedge_attr_t = torch.zeros((hyperedge_index.size(1),self.hyperedge_out_channels),dtype=torch.float32)
        loc = 0
        for key, value in self.target_hyperedge_cantor.items():
            cantor_id = value.unsqueeze(1)
            matches = torch.all(hyperedge_search==cantor_id,dim=0)
            indices = matches.nonzero()[:,0]
            hyperedge_attr_t[indices,loc] = 1
            loc += 1
        hyperedge_attr_t[~torch.any(hyperedge_attr_t==1,dim=1),self.hyperedge_out_channels-1] = 1
        return hyperedge_attr_t

    def get_masks(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor,Tensor]:
        r"""Get node and edge masks for neutrino association.
        Returning node and edge masks.

        Args:
            x (Tensor): Node input features.
            edge_index (Tensor): Edge index tensor.

        :rtype: :class:`Tuple[Tensor,Tensor]`
        """
        x_fw_mask_loc = torch.any(x[:,-1].unsqueeze(1)==self.neutrino_association.unsqueeze(0),
                                  dim=1).nonzero().flatten()
        x_fw_mask = torch.zeros(x.size(0), dtype=torch.int64)
        x_fw_mask[x_fw_mask_loc] = 1
        edge_fw_mask_loc = torch.any(edge_index[1].unsqueeze(1)==x_fw_mask_loc.unsqueeze(0),
                                     dim=1).nonzero().flatten()
        edge_fw_mask = torch.zeros(edge_index[1].size(0), dtype=torch.int64)
        edge_fw_mask[edge_fw_mask_loc] = 1
        return x_fw_mask, edge_fw_mask

    def processing(self, index):
        x = self.build_node_attr(self.file['INPUTS'],index)
        edge_index, edge_attr = self.build_edge_attr(x)
        u = self.build_glob_attr(self.file['INPUTS'],index)
        x_fw_mask, edge_fw_mask = self.get_masks(x, edge_index) if self._use_diffusion else (None, None)
        hyperedge_index = self.build_hyperedge_index(x) if self._use_hyperedge else None

        if self._train_mode is False:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u,
                        x_fw_mask=x_fw_mask, edge_fw_mask=edge_fw_mask,
                        hyperedge_index=hyperedge_index,
                        topo_max_num_edges=torch.tensor([[self.topo_max_num_edges]],dtype=torch.float32),
                        topo_max_num_hyperedges=torch.tensor([[self.topo_max_num_hyperedges]],dtype=torch.float32),
                        topo_num_neutrinos=torch.tensor([[self.topo_num_neutrinos]],dtype=torch.float32))
        else:
            if self._use_edge or self._use_hyperedge:
                node_cantor = self.get_node_cantor_id(self.file['LABELS'],index)
            if self._use_hyperedge:
                hyperedge_cantor = torch.cat([
                    node_cantor[hyperedge_index[i]].unsqueeze(0) for i in range(hyperedge_index.size(0))])
                hyperedge_attr_t = self.build_hyperedge_target(hyperedge_cantor, hyperedge_index)
            else:
                hyperedge_attr_t = None
            if self._use_diffusion:
                neutrino_t = self.build_neutrino_target(self.file['LABELS'],index)
            else:
                neutrino_t = None
            if self._use_edge:
                edge_cantor = torch.cat([node_cantor[edge_index[0]].unsqueeze(0),
                                         node_cantor[edge_index[1]].unsqueeze(0)])
                edge_attr_t = self.build_edge_target(edge_cantor, edge_index)
            else:
                edge_attr_t = None
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u,
                        edge_attr_t=edge_attr_t, neutrino_t=neutrino_t,
                        x_fw_mask=x_fw_mask, edge_fw_mask=edge_fw_mask,
                        hyperedge_index=hyperedge_index, hyperedge_attr_t=hyperedge_attr_t,
                        topo_max_num_edges=torch.tensor([[self.topo_max_num_edges]],dtype=torch.float32),
                        topo_max_num_hyperedges=torch.tensor([[self.topo_max_num_hyperedges]],dtype=torch.float32),
                        topo_num_neutrinos=torch.tensor([[self.topo_num_neutrinos]],dtype=torch.float32))

        data = self.transform(data)
        torch.save(data, osp.join(self.cache, f'processed_{index}.pt'))
        return data

    def __getitem__(self, index) -> Data:
        cached_data = os.path.join(self.cache, f'processed_{index}.pt')
        if osp.exists(cached_data):
            return torch.load(cached_data)
        return self.processing(index)

    def __len__(self):
        return self.size
