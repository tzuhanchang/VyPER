# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Removed

### Fixed


## [0.2.3-alpha] - 2025-11-19

### Added
 - Add learning rate scheduler ([#60](https://github.com/tzuhanchang/VyPER/pull/60))
 - Add input dataset path printout ([#59](https://github.com/tzuhanchang/VyPER/pull/59))

### Changed
 - Change how model checkpoints are saved and loaded ([#63](https://github.com/tzuhanchang/VyPER/pull/63))

### Fixed
 - Fix `pip` version specifiers in MacOS environment file ([#61](https://github.com/tzuhanchang/VyPER/pull/61))


## [0.2.2-alpha] - 2025-11-04

### Added
 - Add more user-configurable options for the diffusion module ([#57](https://github.com/tzuhanchang/VyPER/pull/57))
 - Add missing documentation for the predicting configuration block ([#49](https://github.com/tzuhanchang/VyPER/pull/49))

### Changed
 - Upgrade dependencies: Python 3.12, PyTorch 2.8, CUDA 12.8 ([#51](https://github.com/tzuhanchang/VyPER/pull/51))
 - Allow users to define `num_sampling_steps` independently for training and predicting ([#46](https://github.com/tzuhanchang/VyPER/pull/46))
 - Generalize the network for zero-neutrino topologies ([#45](https://github.com/tzuhanchang/VyPER/pull/45))

### Fixed
 - Fix `num_sampling_steps` is not in struct error in `tune.py` ([#55](https://github.com/tzuhanchang/VyPER/pull/55))
 - Fix runtime error raised after PyTorch upgrade ([#53](https://github.com/tzuhanchang/VyPER/pull/53))
 - Fix reference to CONFIGS in `tune.py` ([#50](https://github.com/tzuhanchang/VyPER/pull/50))
 - Fix `sum` method in `edge_reduction` ([#48](https://github.com/tzuhanchang/VyPER/pull/48))


## [0.2.1-alpha] - 2025-07-04

### Added
 - Add optimizer `torch.optim.SGD` ([#41](https://github.com/tzuhanchang/VyPER/pull/41))
 - Add hyperparameter tuning feature ([#39](https://github.com/tzuhanchang/VyPER/pull/39))

### Fixed
 - Fix negative train/validation split fraction printout ([#42](https://github.com/tzuhanchang/VyPER/pull/42))


## [0.2.0-alpha] - 2025-06-18

### Added
 - Add hyperedge model ([#35](https://github.com/tzuhanchang/VyPER/pull/35))
 - Add hyperedge construction in `VyPER.data.VyPERDataset` ([#29](https://github.com/tzuhanchang/VyPER/pull/29))

### Fixed
 - Fix training for setups with no `target.hyperedge` key ([#36](https://github.com/tzuhanchang/VyPER/pull/36))


## [0.1.1-alpha] - 2025-06-09

### Added
 - Add multi-label edge classification feature ([#31](https://github.com/tzuhanchang/VyPER/pull/31))
 - Add graph caching feature ([#33](https://github.com/tzuhanchang/VyPER/pull/33))

### Changed
 - Use label-independent `x_fw_mask` and `edge_fw_mask` construction ([#27](https://github.com/tzuhanchang/VyPER/pull/27))

### Fixed
 - Fix non-default `edge_reduction` methods ([#30](https://github.com/tzuhanchang/VyPER/pull/30))


## [0.1.0-alpha] - 2025-05-27

### Added
 - Add multiple `ModelCheckpoint` ([#25](https://github.com/tzuhanchang/VyPER/pull/25))
 - Add `AdamW` optimizer ([#23](https://github.com/tzuhanchang/VyPER/pull/23))
 - Add $\Delta R(\nu^\mathrm{truth},\nu^\mathrm{reco})$ log in Tensorboard ([#19](https://github.com/tzuhanchang/VyPER/pull/19))
 - Add neutrino message backward pass ([#18](https://github.com/tzuhanchang/VyPER/pull/18))
 - Add neutrino four-vector definition ([#16](https://github.com/tzuhanchang/VyPER/pull/16))
 - Add `eta` hyperparameter ([#10](https://github.com/tzuhanchang/VyPER/pull/10))
 - Add prediction functionality ([#9](https://github.com/tzuhanchang/VyPER/pull/9))
 - Add configuration file reference ([#8](https://github.com/tzuhanchang/VyPER/pull/8))
 - Add `conda` environment files ([#5](https://github.com/tzuhanchang/VyPER/pull/5))
 - Add `event_filter` option ([#2](https://github.com/tzuhanchang/VyPER/pull/2))

### Changed
 - Replace the Euler sampling method with DDIM ([#21](https://github.com/tzuhanchang/VyPER/pull/21))
 - Relocate environment files ([#7](https://github.com/tzuhanchang/VyPER/pull/7))
 - Update logging strategy ([#4](https://github.com/tzuhanchang/VyPER/pull/4))

### Fixed
 - Fix typo `node_4vector_definition` ([#15](https://github.com/tzuhanchang/VyPER/pull/15))
 - Fix the hardcoded diffusion output size ([#14](https://github.com/tzuhanchang/VyPER/pull/14))
 - Fix the missing `x_fw_mask` and `edge_fw_mask` ([#6](https://github.com/tzuhanchang/VyPER/pull/6))
 - Fix input data not being transformed/scaled ([#3](https://github.com/tzuhanchang/VyPER/pull/3))

[0.2.3-alpha]: https://github.com/tzuhanchang/VyPER/compare/v0.2.2-alpha...v0.2.3-alpha
[0.2.2-alpha]: https://github.com/tzuhanchang/VyPER/compare/v0.2.1-alpha...v0.2.2-alpha
[0.2.1-alpha]: https://github.com/tzuhanchang/VyPER/compare/v0.2.0-alpha...v0.2.1-alpha
[0.2.0-alpha]: https://github.com/tzuhanchang/VyPER/compare/v0.1.1-alpha...v0.2.0-alpha
[0.1.1-alpha]: https://github.com/tzuhanchang/VyPER/compare/v0.1.0-alpha...v0.1.1-alpha
[0.1.0-alpha]: https://github.com/tzuhanchang/VyPER/releases/tag/v0.1.0-alpha
