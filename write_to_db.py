"""Reads the outputs of a watershed + waterz segmentation + agglomeration, and
writes them to a MongoDB database.

We recommend putting the database credentials in a YAML file like below:

```
name: "fatepredict_alice"
user: "username"
pass: "password"
host: "mongodb-server"
port: 12345
```
"""
# TODO
# - [x] Create index
# - [ ] Set cell attributes
# - [ ] Include merge-parent information
# - [ ] edges ?????
import math

import configargparse
import pymongo


def Parser():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', is_config_file=True)
    parser.add('-n', '--name', required=True, help="Project Name")
    parser.add('--host', required=True, help="MongoDB Host Name")
    parser.add('--port', required=True, help="MongoDB Port")
    parser.add('--user', required=True, help="Username")
    parser.add('--pass', required=True, help="Password")
    return parser


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    # Creates client
    client = pymongo.MongoClient(host=args.host)
    db = client[args.name]
    create_indices = 'nodes' not in db.list_collection_names()
    cells = db['nodes']

    # Create indices for quick lookup
    cells.create_index(
        [
            (loc, pymongo.ASCENDING)
            for loc in ['t', 'z', 'y', 'x']
        ],
        name='position')
    cells.create_index(
        [
            ('id', pymongo.ASCENDING)
        ],
        name='id',
        unique=True)

    # Hypotheses:
    """
    voxel_size = batch[maxima].spec.voxel_size
    position = roi.get_begin() + voxel_size*index
    """

    def get_position():
        # TODO
        # position = roi.get_begin() + voxel_size*index
        return (0, 0, 0, 0)

    position = get_position()
    voxel_size = (1, 1, 1)  # TODO

    # Can probably get this from the labels in the fragments
    # TODO make sure it matches index in the database
    cell_id = int(math.cantor_number(position / voxel_size))

    # Cell score
    score = 0  # TODO get score value from zarr/on-the-fly from waterz
    # TODO should we use affine transform information
    movement_vector = tuple(0, 0, 0)  # in just spatial dimensions
    # TODO get merge parent from merge history
    merge_parent = None

    # TODO can we add the merge score to the cell score + of which  cell???
    # Merge score on child = doesn't make sense
    # merge score on parent == multiple merge scores for single parent!

    # INSERT command
    cells.append({
        'id': cell_id,
        'score': float(score),
        't': position[0],
        'z': position[1],
        'y': position[2],
        'x': position[3],
        'movement_vector': movement_vector,
        'merge_parent': merge_parent
    })
