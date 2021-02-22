### Simple Examples

#### Definitions
This is a simple simulation environment based on gym-minigrid to create environment more close to drone fly map <br>
https://github.com/maximecb/gym-minigrid

#### How to import a minigrid
there are three different minigrid available now <br>
```
from gym_minigrid.envs.empty import EmptyEnv
env = EmptyEnv(width=10, height=20, max_step=100)
```
The empty environment is an environment without anything expect the goal, agent and the wall to limit the space. <br>
Agent and Goal will never in the same point at start.

DistShiftEnv and LavaGapEnv is environment with extra Lava, in this project, the Agent can not see anything behind Lava to match the video input when Agent fly.

```
from gym_minigrid.envs.lavagap import LavaGapEnv
from gym_minigrid.envs.distshift import DistShiftEnv
LavaGapEnv(size, max_step=50)
DistShiftEnv(width=size, height=size, max_step=50)
```

#### How to train an agent
To train an agent, try example.py <br>
The default model is a 6 layer MLP <br>
The train algorithm is Deep Q Network, all parameters can be found in the config.py <br>
sxs