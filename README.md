# FatePredict Experiments

Scripts to run experiments for the FatePredict project.

## Setup

Note: the below currently only works on Linux machines.

First, install requirements for `waterz` with your package manager, e.g.
```
sudo apt install libboost-dev
```

Python requirements are listed in `requirements.txt`, and can be installed with `pip`.
We recommend creating a dedicated python environment e.g. with `conda`.

## TODO
- [x] Watershed + Waterz script
- [ ] Extracting edges between time points
- [ ] Writing candidate graph to a Mongo DB
- [ ] Apply `linajea` for tracking
