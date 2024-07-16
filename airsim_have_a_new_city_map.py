from __future__ import annotations

from typing import Any

from environments.CustomGrid import Grid, Lava, StartPoint, BuildTile, RoadTile, DamageTile
from environments.EmptyAndMap import EmptyWithMapEmpty
from minigrid.core.actions import IntEnum
import numpy as np
import itertools
import random
import json

with open("./map.txt", "r") as f:
    whole_map = json.load(f)["map"]
    for each in whole_map:
        obj_type = ID_TO_OBJ[each["type"]]
        if obj_type is not None:
            obj = obj_type()
            obj.unity_pos = [each["pos_x"], each["pos_y"]]
            self.whole_grid.set(each["x"], each["y"], obj)
        self.whole_grid_start = np.array([-322.5, -324.5])
