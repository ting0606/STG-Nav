import habitat
import habitat_sim
import numpy as np
from PIL import Image

# 引入其他必要模組
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# 初始化 GLIP 模型
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "glip_tiny_model_o365_goldg_cc_sbu.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

test_scene = "/home/ting/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}
sim_settings = {
        "width": 256,  # Spatial resolution of the observations，分辨率
        "height": 256,
        "scene": test_scene,  # Scene path
        "default_agent": 0, # 设置默认agent为0
        "sensor_height": 1.5,  # Height of sensors in meters，高度
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation，设置伪随机数字再生种子
        "enable_physics": False,  # kinematics only，是否需要启动交互性，对于导航任务不需要，对于重排任务需要
}

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

def glip_inference(image_, caption_):
    preds = glip_demo.compute_prediction(image_, caption_)
    top_preds = glip_demo._post_process(preds, threshold=0.5)

    labels = top_preds.get_field("labels").tolist()
    scores = top_preds.get_field("scores").tolist()
    boxes = top_preds.bbox.detach().cpu().numpy()
    labels_names = glip_demo.get_label_names(labels)

    return boxes, scores, labels_names

# Habitat-Sim 配置
def create_habitat_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration() # 获取一个全局配置的结构体
    sim_cfg.gpu_device_id = 0 # 在0号gpu上进行配置
    sim_cfg.scene_id = sim_settings["scene"] # 场景设置
    sim_cfg.enable_physics = sim_settings["enable_physics"] # 物理功能是否启用，默认情况下为false

    # Note: all sensors must have the same resolution
    sensor_specs = []
	
	# 创建实例然后填充传感器配置参数
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor" # uuid必须唯一
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
    color_sensor_spec.position = [0.0, sim_settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec) # 将color_sensor_spec加入到sensor_specs中

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
    depth_sensor_spec.position = [0.0, sim_settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec) # 将sensor_specs加入到sensor_specs中

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
    semantic_sensor_spec.position = [0.0, sim_settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec) # 将semantic_sensor_spec加入到sensor_specs中

    # Here you can specify the amount of displacement in a forward action and the turn angle
    # 有了传感器后必须将其加到agent上
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    # 定义agent的观测空间
    # 给agent配置所有的传感器
    agent_cfg.sensor_specifications = sensor_specs
    # 定义agent的动作空间
    # agent可以做到的动作，是一个字典包含前进、左转和右转
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25) # 向前进0.25m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0) # 向左转30度
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0) # 向右转30度
        ),
    }
	# 返回habita的全局配置，sim_cfg环境配置，以及agent的配置[agent_cfg]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


if __name__ == '__main__':
    sim = create_habitat_sim()
    
    # 獲取環境中的圖像
    observations = sim.get_sensor_observations()
    if "color_sensor" not in observations:
            raise KeyError("Sensor 'color_sensor' is missing in the observations. Check the configuration.")
        
        # Access RGB image from observations
    rgb_image = observations["color_sensor"]
    
    # 轉換圖像為 PIL 格式
    image = Image.fromarray(rgb_image[:, :, :3])
    
    caption = "A chair and a table in the room"
    boxes, scores, labels_names = glip_inference(np.array(image), caption)

    print("Detected objects:", labels_names)
    print("Scores:", scores)
    print("Bounding boxes:", boxes)
    
    # 可選：將結果可視化
    draw_images(image=image, boxes=boxes, classes=labels_names, scores=scores, colors=Colors())

    sim.close()
