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

We recomand using conda env by running the following command:
```
conda create --name fatepredict python pytorch pylp -c pytorch -c funkey
conda activate fatepredict
cd <your clone repo dir>
pip install -r fatepredict_experiments/requirements.txt 
```

## TODO
- [x] Watershed + Waterz script
- [ ] Extracting edges between time points
- [ ] Writing candidate graph to a Mongo DB
- [ ] Apply `linajea` for tracking


## mango DB 
MangoDB database for linux [Download :link:](https://www.mongodb.com/docs/manual/administration/install-community/)
Set a local `db` [VSCODE extension :link:](https://code.visualstudio.com/docs/azure/mongodb)

## Zarr structure

* Raw (c,t,z,y,x)
* Groundtruth_segmentation (t,z,y,x)
* Fragment (t,z,y,x) -> Each fragment is unique
* Merge tree `<per frame>`
    * Merge (n,3) (u,v,w) -> u and v merge to w
    * Scoring (n,) 
* Fragment_stats `<per frame>`
    * id (n,) -> fragment id
    * Position (n,3) -> (z,y,x)
    * Volume -> fragment region size
    * Overlap count and pair list
* Groundtruth tracks
    * Node_IDs (n) -> unique id for node
    * Node_position (n,3) -> (z,y,x)
    * Node_parent -> id of node parent
* gt_node (n,2) -> (id,label) node `id` has `label`
* Transfermations `<per frame>` 4 by 4 affine matrix
* Tracks :star:
    * id -> segments' id
    * parents