from gym_minigrid.minigrid import Lava as la


class Lava(la):
    def __init__(self):
        super().__init__()

    def see_behind(self):
        return False
