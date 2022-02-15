import os
import mxnet as mx
from mxnet import nd
from mpu.ml import indices2one_hot

import numpy as np

agent_dir = {
    0: '>',
    1: 'V',
    2: '<',
    3: '^',
}


def rew_clipper(rew):
    if rew > 0.:
        return 1.
    elif rew < 0.:
        return -1.
    else:
        return 0


def preprocess(raw_frame, image_size, channel=3, frame_len=4, current_state=None, initial_state=False):
    raw_frame = nd.array(raw_frame, mx.cpu())
    if channel == 1:
        raw_frame = nd.mean(raw_frame, axis=2)
        raw_frame = nd.reshape(raw_frame, shape=(raw_frame.shape[0], raw_frame.shape[1], 1))
    raw_frame = mx.image.imresize(raw_frame, image_size, image_size)
    raw_frame = nd.transpose(raw_frame, (2, 0, 1))
    raw_frame = raw_frame.astype(np.float32) / 255
    if frame_len > 1:
        if initial_state:
            state = raw_frame
            for _ in range(frame_len - 1):
                state = nd.concat(state, raw_frame, dim=0)
        else:
            state = mx.nd.concat(current_state[raw_frame.shape[0]:, :, :], raw_frame, dim=0)
        return state, raw_frame
    else:
        return raw_frame, raw_frame

def to_one_hot(array, classes):
    shape = list(array.shape) + [-1]
    array = array.flatten()
    return np.array(indices2one_hot(array, nb_classes=classes + 1)).reshape(shape)


def translate_gym(data):
    return {"memory": [i[0] for i in data], "battery": [i[1] for i in data]}


# return to normal
def translate_state(state):
    _state = translate_gym(state)
    return _state["memory"], _state["battery"]
    # return state["agent_view"], state["whole_map"], state["memory"], state["battery"]


def copy_params(offline, online):
    layer = list(offline.collect_params().values())
    for i in layer:
        _1 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()
        online.collect_params().get("_".join(i.name.split("_")[1:])).set_data(
            i.data())
        _2 = online.collect_params().get(
            "_".join(i.name.split("_")[1:])).data().asnumpy()


def check_dir(i):
    # create required path
    if not os.path.exists("./{}/".format(i)):
        os.mkdir("./{}/".format(i))


def get_goal(array, agent):
    _min = 999
    _location = np.zeros([array.shape[0], array.shape[1]])
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            if value in (3, 4):
                _dis = sum(np.abs(np.array(agent[:2]) - np.array((i, j))))
                if _dis < _min:
                    _location = np.zeros([array.shape[0], array.shape[1]])
                    _location[i][j] = 1
                    _min = _dis
    return _location


def to_numpy(grid, allow):
    """
    Produce a pretty string of the environment's grid along with the agent.
    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.
    """
    map_img = [[] for i in range(len(allow) + 1)]
    for i in grid.grid:
        map_img[0].append(0)
        if i is not None and i.type in allow.keys():
            map_img[allow[i.type]].append(1)
            _allow = set(allow.values())
            _allow.remove(allow[i.type])
            for j in _allow:
                map_img[j].append(0)
        else:
            for j in set(allow.values()):
                map_img[j].append(0)
    return np.array(map_img).reshape((-1, grid.width, grid.height))


def create_input(data):
    target = []
    for i in range(len(data[0])):
        target.append([])
    for arg in data:
        _ = [np.array([i]) if type(i) is int else np.array(i) for i in arg]
        for i, each in enumerate(_):
            target[i].append(each)
    target = [np.array(i) for i in target]
    return target
