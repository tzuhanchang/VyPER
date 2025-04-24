# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Removed

### Fixed
 - Fix typo `node_4vector_definition` ([#15](https://github.com/tzuhanchang/VyPER/pull/15))
 - Fix the hardcoded diffusion output size ([#14](https://github.com/tzuhanchang/VyPER/pull/14))
 - Fix the missing `x_fw_mask` and `edge_fw_mask` ([#6](https://github.com/tzuhanchang/VyPER/pull/6))
 - Fix input data not being transformed/scaled ([#3](https://github.com/tzuhanchang/VyPER/pull/3))
