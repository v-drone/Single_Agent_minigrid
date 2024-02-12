import re
import mlflow
import numpy as np
import onnxruntime as ort
from IPython import display
import matplotlib.pyplot as plt
from utils import minigrid_env_creator

# Set mlflow
mlflow.set_tracking_uri("http://10.56.238.20:9999")
mlflow.set_experiment(experiment_name="RouteWithTrod")
mlflow_client = mlflow.tracking.MlflowClient()
for_pred = "./for_pred/"
run_id = "6ebe60216fda449c9e58d21c7392080d"
artifacts = mlflow_client.list_artifacts(run_id)
artifacts = [file_info for file_info in artifacts if not re.match(r'\d+.json', file_info.path)]
for each in artifacts:
    mlflow_client.download_artifacts(run_id=run_id, path=each.path, dst_path=for_pred)
session = ort.InferenceSession("./for_pred/wrapped_model.onnx")

env_config = {
    "id": "Route",
    "size": 12,
    "routes": (2, 4),
    "max_steps": 300,
    "battery": 100,
    "img_size": 100,
    "tile_size": 8,
    "num_stack": 30,
    "render_mode": "rgb_array",
    "agent_pov": False
}
env_example = minigrid_env_creator(env_config)
actions = []
obs, _ = env_example.reset()

while True:
    outputs = session.run(["advantage", "value", "logit"],
                          {"obs": np.expand_dims(obs, axis=0).astype(np.float32)})
    action = np.argmax(outputs[0])
    actions.append(action)
    obs, reward, terminated, truncated, info = env_example.step(action)
    plt.imshow(env_example.render())
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if terminated or truncated:
        break
