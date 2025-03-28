Configuration File Reference
=======================

Dataset, network architecture, training, and predicting are configured with a `.yaml` configuration file.

You could create your configuration files and place them in the `configs` directory.
You can load your customized configuration files using the `--config-name` argument, for example:
```
python -m HyPER.train --config-name=my_config
```
To learn more about VyPER's configuration framework, check out [Hydra](https://hydra.cc/docs/intro/#basic-example).

The configuration file is divided into two sections:

- [Common settings](#common-settings)
- [Dataset-dependent settings](#dataset-dependent-settings)


Common settings
------------------

These settings can be overwritten in the command line:
```
python -m VyPER.train training.learning_rate=0.001
```
overwrites the default [training.learning_rate](#traininglearning_rate) to `0.001`.

- [datasets](#datasets)

    * [datasets.train_set](#datasetstrain_set)
    * [datasets.val_set](#datasetsval_set)
    * [datasets.predict_set](#datasetspredict_set)
    * [datasets.event_filter](#datasetsevent_filter)
    * [datasets.train_val_split](#datasetstrain_val_split)
    * [datasets.drop_last](#datasetsdrop_last)

- [network](#network)

    * [network.message_feats](#networkmessage_feats)
    * [network.num_attn_heads](#networknum_attn_heads)
    * [network.num_message_layers](#networknum_message_layers)
    * [network.hyperedge_feats](#networkhyperedge_feats)
    * [network.hyperedge_order](#networkhyperedge_order)
    * [network.num_sampling_steps](#networknum_sampling_steps)

- [training](#training)

    * [training.learning_rate](#traininglearning_rate)
    * [training.optimizer](#trainingoptimizer)
    * [training.loss_reduction](#trainingloss_reduction)
    * [training.alpha](#trainingalpha)
    * [training.eta](#trainingeta)
    * [training.dropout](#trainingdropout)
    * [training.gradient_clip](#traininggradient_clip)
    * [training.epochs](#trainingepochs)
    * [training.batch_size](#trainingbatch_size)
    * [training.grad_accum_batches](#traininggrad_accum_batches)
    * [training.patience](#trainingpatience)
    * [training.save_directory](#trainingsave_directory)
    * [training.continue_from_ckpt](#trainingcontinue_from_ckpt)

- [predicting](#predicting)

    * [predicting.model_directory](#predictingmodel_directory)
    * [predicting.batch_size](#predictingbatch_size)
    * [predicting.edge_reduction](#predictingedge_reduction)
    * [predicting.save_as](#predictingsave_as)

- [device](#device)

    * [device.accelerator](#deviceaccelerator)
    * [device.num_devices](#devicenum_devices)
    * [device.num_workers](#devicenum_workers)


## datasets

Configurations of the datasets to be used.


### datasets.train_set

The path to the training HDF5 dataset.

Required for network training.


### datasets.val_set

The path to the validation HDF5 dataset.

Required for network training.


### datasets.predict_set

The path to the prediction HDF5 dataset.

Required for prediction.


### datasets.event_filter

Name of a boolean vector saved in the dataset, e.g. `METADATA_FullyMatched`.

As a result, VyPER will only use the events marked true in the `METADATA/FullyMatched` vector.


### datasets.train_val_split

The training/validation ratio splits `train_set` into a training and validation dataset.

This option only becomes active when only the `train_set` is provided.


### datasets.drop_last

Set to `True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
If `False` and the size of the dataset is not divisible by the batch size, then the last batch will be smaller.


## network

Configurations of the network architecture.


### network.message_feats

Message embedding size during message-passing operation.


### network.num_attn_heads

Number of attention heads in the attention layers.


### network.num_message_layers

Number of message-passing layers.


### network.hyperedge_feats

Hyperedge embedding size.


### network.hyperedge_order

Order of the hyperedges, $|\tilde{E}_m|$.
e.g. top quark has three final states, therefore, `hyperedge_order=3`.


### network.num_sampling_steps

Number of diffusion ODE sampling steps.


## training

Configurations for the network training.


### training.learning_rate

Learning rate.


### training.optimizer

Gradient descent algorithm.

Supported optimizers:
 - `Adam`: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).


### training.loss_reduction

Specifies the reduction to apply to the loss function output.

Supported methods:
 - `mean`: average of the event loss in the batch. (Recommended, independent of `batch_size`)
 - `sum`: sum of the event loss in the batch.
 - `min`: smallest event loss in the batch.
 - `max`: highest event loss in the batch.


### training.alpha

Mixing fraction, $\alpha$, of edge and hyperedge loss.

The final loss is computed according to: $\mathcal{L} = \eta\mathcal{L}_d+(1-\eta)\[ \alpha \mathcal{L}_h + (1-\alpha)\mathcal{L}_e \]$, where $\mathcal{L}_d$ is the diffusion loss, $\mathcal{L}_h$ is the hyperedge loss and $\mathcal{L}_e$ is the edge loss.


### training.eta

Mixing fraction, $\eta$, of diffusion-to-reconstruction loss.

The final loss is computed according to: $\mathcal{L} = \eta\mathcal{L}_d+(1-\eta)\[ \alpha \mathcal{L}_h + (1-\alpha)\mathcal{L}_e \]$, where $\mathcal{L}_d$ is the diffusion loss, $\mathcal{L}_h$ is the hyperedge loss and $\mathcal{L}_e$ is the edge loss.


### training.dropout

Probability of an element to be zeroed in the Dropout.


### training.gradient_clip

The value to clip gradients' global norm.

Gradient clipping can be enabled to avoid exploding gradients.
Set to `0` to have no gradient clipping.


### training.epochs

Maximum number of epochs.

Training can be terminated before reaching the maximum number of epochs due to the implementation of the early stopping, see [training.patience](#trainingpatience).


### training.batch_size

The number of training examples used in one iteration or backward pass (when [training.grad_accum_batches](#traininggrad_accum_batches) is set to `1`) through the network.


### training.grad_accum_batches

Number of small batches ($\mathrm{K}$) before doing a backward pass.

The effect is a large effective batch size of size $\mathrm{K\times N}$, where $\mathrm{N}$ is the batch size.
This is useful when GPU memory is limited.
Set to `1` to have no accumulated gradients.


### training.patience

Number of unconverged epochs to wait before terminating training.


### training.save_directory

Location where the training states are saved.


### training.continue_from_ckpt

Path where the training state is loaded and continuing training.

Set to `null` not loading any checkpoints, and start a new training.


## device

Configurations of the hardware to be used for training and reconstruction.

### device.accelerator

Type of accelerator to be used for training or prediction.

Supported options:
 - `cpu`: CPU accelerator.
 - `gpu`: CUDA&trade; supported GPU(s).


### device.num_devices

Number of CPU or GPU devices to be used for training or prediction.


### device.num_workers

Number of subprocesses to use for data loading.

Set to `0` means that the data will be loaded in the main process.
A good starting point is to set it to the number of CPU cores on the machine. However, the larger the `num_workers`, the more CPU memory consumed.



Dataset-dependent settings
------------------

These settings depend on the dataset provided.
They are not (easily) overwritable in the command line.

This section will use the sample $t\bar{t}$ dilepton final-state dataset as an example to explain each setting.


- [input](#input)

    * [input.nodes](#inputnodes)
    * [input.node_features](#inputnode_features)
    * [input.node_4vector_definition](#inputnode_4vector_definition)
    * [input.node_transforms](#inputnode_transforms)
    * [input.edge_features](#inputedge_features)
    * [input.edge_transforms](#inputedge_transforms)
    * [input.global_features](#inputglobal_features)
    * [input.global_transforms](#inputglobal_transforms)

- [target](#target)

    * [target.edge](#targetedge)
    * [target.neutrinos](#targetneutrinos)
    * [target.hyperedge](#targethyperedge)


## input

Dataset configurations for network inputs.

### input.nodes

```yaml
input:
  ...
  nodes:
    JET: 0
    ELECTRON: 1
    MUON: 2
  ...
```
Names of the final-state objects stored in the `INPUTS` data group will be used as nodes.
The integer associated with each object is a user-assigned ID used for node type identification.


### input.node_features

```yaml
input:
  ...
  node_features:
    - e
    - eta
    - phi
    - pt
    - btag
    - charge
  ...
```
Node input features.

These should match the `dtype.names` of the `INPUTS/JET`, `INPUTS/ELECTRON` and `INPUTS/MUON` datasets.


### input.node_4vector_definition

```yaml
input:
  ...
  node_4vector_definition:
    functional: MomentumTensor.EEtaPhiPt
    ordered_inputs:
      - e
      - eta
      - phi
      - pt
  ...
```
Node four-momentum definition.

Node four-momentum is defined using the class set in `functional`.
This class is initiated with an ordered set of node features, defined in `ordered_inputs`.


### input.node_transforms

```yaml
input:
  ...
  node_transforms:
    - torch.log(x)
    - x / math.pi
    - x / math.pi
    - torch.log(x)
    - x
    - x
  ...
```
Transformation methods for the node input features.

Each node feature is transformed/scaled according to the methods defined in `node_transforms`.
These methods are arranged in the same order as [input.node_features](#inputnode_features).


### input.edge_features

```yaml
input:
  ...
  edge_features:
    - (e1.eta - e2.eta)
    - ...
    - ...
    - (e1 + e2).m
  ...
```
Edge input features.

Edge features are computed using the four-momentum of its endpoint nodes (`e1` and `e2`).
Please check the class methods of the `functional` in [input.node_4vector_definition](#inputnode_4vector_definition).


### input.edge_transforms

```yaml
input:
  ...
  edge_transforms:
    - x
    - x
    - x
    - torch.log(x)
  ...
```
Transformation methods for the edge input features.

Each edge feature is transformed/scaled according to the methods defined in `edge_transforms`.
These methods are arranged in the same order as [input.edge_features](#inputedge_features).


### input.global_features

```yaml
input:
  ...
  global_features:
    - njet
    - nbTagged
    - nelectron
    - nmuon
    - met_met
    - met_phi
  ...
```
Global input features.

These should match the `dtype.names` of the `INPUTS/GLOBAL` datasets.


### input.global_transforms

```yaml
input:
  ...
  global_transforms:
    - x / 2
    - x / 2
    - x / 2
    - x / 2
    - torch.log(x)
    - x
  ...
```
Transformation methods for the global input features.

Each global feature is transformed/scaled according to the methods defined in `global_transforms`.
These methods are arranged in the same order as [input.global_features](#inputglobal_features).


## target

Dataset configurations for targets.

In the following sections, you will encounter representations where two integers are separated by a hyphen, such as `'0-1'`.
These are referred to as "object labels".
The first integer refers to the object ID defined in [input.nodes](#inputnodes).
The second integer is the final-state truth label.
In the sample $t\bar{t}$ dilepton final-state dataset, the final states are labeled in the following way:

`LABELS/JET`:
| | $b$ | $\bar{b}$ | Other | 
| ------------- | ------------- | ------------- | ------------- |
| ID | 1 | 4 | 0 |

`LABELS/ELECTRON` or `LABELS/MUON`:
| | $l^+$ | $l^-$ |
| ------------- | ------------- | ------------- |
| ID | 1 | 2 |

Therefore, `'0-1'` represents the true $b$ jet from the top quark, `'2-2'` represents the $\mu^-$ from the $W^-$ boson of the anti-top quark, and so on.


### target.edge

```yaml
target:
  ...
  edge:
    bl:
      - ['0-1','1-1'] # b el+
      - ['0-1','2-1'] # b mu+
      - ['0-4','1-2'] # bbar el-
      - ['0-4','2-2'] # bbar el-
    ...
  ...
```
Definitions of target edges.

Each target edge is represented as a list containing two endpoint-nodes of the edge. `bl` is the class name, currently only one edge class is supported.


### target.neutrinos

```yaml
target:
  ...
  neutrinos:
    node_label: 3
    associated_nodes:
      - 1-2 # el-
      - 2-2 # mu-
      - 1-1 # el+
      - 2-1 # mu+
    features:
      - px
      - py
      - pz
    transforms:
      - (x + 0.0179) / 53.3342
      - (x + 0.0069) / 53.3252
      - (x + 0.0306) / 160.5729
    reverse_transforms:
      - x * 53.3342 - 0.0179
      - x * 53.3252 - 0.0069
      - x * 160.5729 - 0.0306
  ...
```
Definitions of target neutrinos.

`node_label`: an integer assigned to the reconstructed neutrino nodes for node type identification.

`features`: neutrino target features. These should match the `dtype.names` of the `LABELS/NEUTRINO` datasets.

`transforms`: transformation methods for the neutrino target features.
Each neutrino target feature is transformed/scaled accordingly.
These methods are arranged in the same order as `features`.

`reverse_transforms`: neutrino predictions are transformed back to the expected states accordingly.


### target.hyperedge

```yaml
target:
  ...
  hyperedge:
    top:
      - ['0-1','1-2','3-1-2']
      - ['0-1','2-2','3-2-2']
      - ['0-2','1-1','3-1-1']
      - ['0-2','2-1','3-2-1']
  ...
```
Definitions of target hyperedges.

Each target hyperedge is represented as a list containing three or more nodes. `top` is the class name; currently, only one hyperedge class is supported.
