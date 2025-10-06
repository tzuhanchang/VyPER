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
- [Tuning settings](#tuning-settings)
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
    * [datasets.tune_set](#datasetstune_set)
    * [datasets.event_filter](#datasetsevent_filter)
    * [datasets.cache_dir](#datasetscache_dir)
    * [datasets.force_reload](#datasetsforce_reload)
    * [datasets.train_val_split](#datasetstrain_val_split)
    * [datasets.drop_last](#datasetsdrop_last)

- [network](#network)

    * [network.message_feats](#networkmessage_feats)
    * [network.num_attn_heads](#networknum_attn_heads)
    * [network.num_message_layers](#networknum_message_layers)
    * [network.hyperedge_feats](#networkhyperedge_feats)
    * [network.hyperedge_order](#networkhyperedge_order)

- [training](#training)

    * [training.learning_rate](#traininglearning_rate)
    * [training.weight_decay](#trainingweight_decay)
    * [training.momentum](#trainingmomentum)
    * [training.optimizer](#trainingoptimizer)
    * [training.loss_reduction](#trainingloss_reduction)
    * [training.alpha](#trainingalpha)
    * [training.eta](#trainingeta)
    * [training.dropout](#trainingdropout)
    * [training.gradient_clip](#traininggradient_clip)
    * [training.epochs](#trainingepochs)
    * [training.batch_size](#trainingbatch_size)
    * [training.num_sampling_steps](#trainingnum_sampling_steps)
    * [training.grad_accum_batches](#traininggrad_accum_batches)
    * [training.patience](#trainingpatience)
    * [training.save_directory](#trainingsave_directory)
    * [training.continue_from_ckpt](#trainingcontinue_from_ckpt)

- [predicting](#predicting)

    * [predicting.model_directory](#predictingmodel_directory)
    * [predicting.model_choice](#predictingmodel_choice)
    * [predicting.batch_size](#predictingbatch_size)
    * [predicting.num_sampling_steps](#predictingnum_sampling_steps)
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


### datasets.tune_set

The path to the hyperparameter tuning HDF5 dataset.

Required for hyperparameter tuning.


### datasets.event_filter

Name of a boolean vector saved in the dataset, e.g. `METADATA_FullyMatched`.

As a result, VyPER will only use the events marked true in the `METADATA/FullyMatched` vector.


### datasets.cache_dir

Path of where the processed graphs are cached.

If it is not provided, by default, VyPER will save the processed graphs in the `.cache` folder.


### datasets.force_reload

If set to `true`, VyPER will clean up the cached graphs and force all the graphs to be processed again.


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


## training

Configurations for the network training.


### training.learning_rate

Learning rate.


### training.weight_decay

Weight decay coefficient. This parameter is valid only if `AdamW` is chosen as the optimizer (see [training.optimizer](#trainingoptimizer)).


### training.momentum

Momentum factor. This parameter is valid only if `SGD` is chosen as the optimizer (see [training.optimizer](#trainingoptimizer)).


### training.optimizer

Gradient descent algorithm.

Supported optimizers:
 - `Adam`: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
 - `AdamW`: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
 - `SGD`: [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf).


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


### training.num_sampling_steps

Number of diffusion ODE sampling steps used for logging during training.


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


## predicting

Configurations of dataset evaluation.


### predicting.model_directory

Path to where the model of choice is saved.

By default, the model directory is the version folder (e.g. "version_0") inside of the `training.save_directory`.
Required for predicting step.


### predicting.model_choice

Choose which saved model state to use.

During the training, top five models with the smallest validation loss and the smallest neutrino $\Delta R$ are saved.
This option give user options to load their preferred model.

Supported options:
 - `min-loss`: model checkpoint with the smallest validation loss.
 - `min-dR`: model checkpoint with the smallest validation $\Delta R$.


### predicting.batch_size

The number of examples used in one predicting iteration.


### predicting.num_sampling_steps

Number of diffusion ODE sampling steps used for logging during predicting.


### predicting.edge_reduction

Method to reduce the two directed edges connecting two endpoints to an undirected one.

Supported options:
 - `mean`: the average of the two directed edges.
 - `max`: the maximum of the two direccted edges.
 - `min`: the minimum of the two directed edges.
 - `sum`: the element-wise sum of the two directed edges.


### predicting.save_as

Location and file name of which the prediction results are saved.

The results are saved in a HDF5 file.
If a file is found at `predicting.save_as`, VyPER will not overwrite the file but throws an error.


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



Tuning settings
------------------

Similar to the [common settings](#common-settings), tuning settings can be overwritten in the command line:
```
python -m VyPER.train tune.n_trials=50
```
it overwrites the default [tune.n_trials](#tunen_trials) to 50.

The hyperparameter tuning is performed using the events in the [`datasets.tune_set`](#datasetstune_set).


- [tune](#tune)

    * [tune.n_trials](#tunen_trials)
    * [tune.epochs](#tuneepochs)
    * [tune.study_name](#tunestudy_name)
    * [tune.save_dir](#tunesave_dir)
    * [tune.sqlite](#tunesqlite)
    * [tune.monitors](#tunemonitors)
    * [tune.directions](#tunedirections)
    * [tune.hyperparameters](#tunehyperparameters)


## tune

Configurations for network fine-tuning with [Optuna](https://optuna.org).

### tune.n_trials

The number of trials for the tuning study. The tuning study continues to create trials until the number of trials reaches  `tune.n_trials`.

Set `tune.n_trials=null` with no limit in terms of the number of trials.

### tune.epochs

The number of epochs performed in each trial.

### tune.study_name

The name of the tuning study.

### tune.save_dir

Location of where the tuning trial states are saved.

### tune.sqlite

The location of the `SQLite` database of where the tuning results are saved.

### tune.monitors

A list of `lightning.Trainer.callback_metrics` that are used to monitor the tuning study.

Available callback metrics:
 - `validation_loss`: The overall network validation loss.
 - `validation_edge_loss`: The edge validation loss.
 - `validation_diffusion_loss`: The diffusion validation loss.
 - `validation_hyperedge_loss`: The hyperedge validation loss (available only if the hyperedge is enabled).
 - `accuracy/edge_channel_*`: The validation edge accuracy of the channel `*`.
 - `accuracy/hyperedge_channel_*`: The validation hyperedge accuracy of the channel `*`.
 - `accuracy/dR_mean`: The validation $\Delta R$ between truth and reconstructed neutrinos.

### tune.directions

Directrions of [tune.monitors](#tunemonitors), which the study is trying to optimise. It can be either `maximize` or `minimize`.

### tune.hyperparameters

A list of hyperparameters to be optimised by the tuning study.

Example:
```yaml
tune:
  hyperparameters:
    network:
      message_feats:
        - int  # Type
        - 32   # Lower bound
        - 128  # Upper bound
        - 16   # Step (optional)
    ...
```
In the above the example, hyperparameter [`network.message_feats`](#networkmessage_feats) will be tuned. The optimised value is an integer and to be found between 32 and 128 with an interval of 16. Almost all [network](#network) and [training](#training) settings can be tuned.



Dataset-dependent settings
------------------

These settings depend on the dataset provided.
They are not (easily) overwritable in the command line.
Configuration blocks labelled [Optional] are omittable, if they are not provided, their corresponding network modules will not be activated.

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

    * [target.topology](#targettopology)
    * [target.edge](#targetedge)
    * [target.neutrinos [Optional]](#targetneutrinos)
    * [target.hyperedge [Optional]](#targethyperedge)


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


### target.topology

```yaml
target:
  ...
  topology:
    edge: 4
    hyperedge: 0
    neutrinos: 2
  ...
```

Definitions of target topology:
  - `edge`: maximum number of target edges could exist in a graph,
  e.g. $2\times 2$ in the example dileptonic $t\bar{t}$ event - 2 bi-directional target `bl` edges.
  - `hyperedge`: maximum number of target hyperedges could exist in a graph.
  - `neutrinos`: number of neutrinos in a graph.

These values are only applicable to the training, they are used for the calculation of the weighted loss.

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

Each target edge is represented as a list containing two endpoint-nodes of the edge.
`bl` is an edge type, multiple edge types are now supported (see [#31](https://github.com/tzuhanchang/VyPER/pull/31)).


### target.neutrinos

This is an optional block, if it is not provided, neutrino diffusion module will not be used.

```yaml
target:
  ...
  neutrinos:
    node_label: 3
    associated_nodes:
      - 1
      - 2
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
    4vector_definition:
      functional: MomentumTensor
      ordered_inputs:
        - torch.sqrt((px*px)+(py*py)+(pz*pz))
        - px
        - py
        - pz
  ...
```
Definitions of target neutrinos.

`node_label`: an integer assigned to the reconstructed neutrino nodes for node type identification.

`associated_nodes`: an integer (see [input.nodes](#inputnodes)) refers to the node(s) which the neutrino is associated with.

`features`: neutrino target features. These should match the `dtype.names` of the `LABELS/NEUTRINO` datasets.

`transforms`: transformation methods for the neutrino target features.
Each neutrino target feature is transformed/scaled accordingly.
These methods are arranged in the same order as `features`.

`reverse_transforms`: neutrino predictions are transformed back to the expected states accordingly.

`4vector_definition`: similar to [`input.node_4vector_definition`](#inputnode_4vector_definition). This allows the results to be correctly logged.


### target.hyperedge

This is an optional block, if it is not provided, hyperedge module will not be used.

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
