from collections import deque, defaultdict
import os
import logging
import time
import json
import torch.nn as nn
import torch
import numpy as np
import networkx as nx
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)

import torch.nn.functional as F
from skimage import measure
import skimage.morphology
import cv2
from model import Semantic_Mapping
from envs.utils.fmm_planner import FMMPlanner
from envs import make_vec_envs
from arguments import get_args
import algo
from PIL import Image

from constants import category_to_id, hm3d_category, category_to_id_gibson,GLIP_category
import envs.utils.pose as pu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glip_predict import GLIPPredictor
from STG import STGBuilder
from openai import OpenAI
import re
import io
import base64
import torchvision.ops as ops 
import json
import re
from scipy.spatial import cKDTree
from skimage.draw import line as line_1


client = OpenAI(api_key="")

os.environ["OMP_NUM_THREADS"] = "1"


fileName = 'data/matterport_category_mappings.tsv'
config_file = "GLIP/glip_Swin_L.yaml" 
weight_file = "GLIP/glip_large_model.pth"

text = ''
lines = []
items = []
hm3d_semantic_mapping={}
hm3d_semantic_index={}
hm3d_semantic_index_inv={}

with open(fileName, 'r') as f:
    text = f.read()
lines = text.split('\n')[1:]

for l in lines:
    items.append(l.split('    '))

for i in items:
    if len(i) > 3:
        hm3d_semantic_mapping[i[2]] = i[-1]
        hm3d_semantic_index[i[-1]] = int(i[-2])
        hm3d_semantic_index_inv[int(i[-2])] = i[-1]


def find_big_connect(image):
    img_label, num = measure.label(image, connectivity=2, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
    # print("img_label.shape: ", img_label.shape) # 480*480
    resMatrix = np.zeros(img_label.shape)
    tmp_area = 0
    for i in range(0, len(props)):
        if props[i].area > tmp_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix = tmp
            tmp_area = props[i].area 
    
    return resMatrix
    

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)
    step_masks = torch.zeros(num_scenes).float().to(device)

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))

    episode_sem_frontier = []
    episode_sem_goal = []
    episode_loc_frontier = []
    for _ in range(args.num_processes):
        episode_sem_frontier.append([])
        episode_sem_goal.append([])
        episode_loc_frontier.append([])

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_process_rewards = 0
    g_total_rewards = np.ones((num_scenes))
    g_sum_rewards = 1
    g_sum_global = 1

    stair_flag = np.zeros((num_scenes))
    clear_flag = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size # 2400/5=480
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    local_ob_map = np.zeros((num_scenes, local_w,
                            local_h))

    local_ex_map = np.zeros((num_scenes, local_w,
                            local_h))

    target_edge_map = np.zeros((num_scenes, local_w,
                            local_h))
    target_point_map = np.zeros((num_scenes, local_w,
                            local_h))


    # dialate for target map
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)
    old_lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    frontier_score_list = []
    for _ in range(args.num_processes):
        frontier_score_list.append(deque(maxlen=10))

    frontier_score_dict = []
    for _ in range(args.num_processes):
        frontier_score_dict.append(deque(maxlen=10))

    G_topo_list = [nx.Graph() for _ in range(num_scenes)]
    region_visit_counter = [defaultdict(int) for _ in range(num_scenes)]
    ini_r = False
    visited_nodes_full = set()
    builder = STGBuilder()

    object_norm_inv_perplexity = torch.tensor(np.load('data/object_norm_inv_perplexity.npy')).to(device)

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def get_frontier_boundaries(frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes

        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
 
        return [int(gx1), int(gx2), int(gy1), int(gy2)]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        print(f"🧹 初始化地圖與姿態 for agent {e}")
        sem_map_module.reset_internal_state()
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        local_map[e].fill_(0.)
        local_ob_map[e]=np.zeros((local_w,
                            local_h))
        local_ex_map[e]=np.zeros((local_w,
                            local_h))
        target_edge_map[e]=np.zeros((local_w,
                            local_h))
        target_point_map[e]=np.zeros((local_w,
                            local_h))

        step_masks[e]=0
        stair_flag[e] = 0
        clear_flag[e] = 0


        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()
        print("🗺️ 地圖與姿態初始化完成 for agent", e)

    init_map_and_pose()


    def remove_small_points(local_ob_map, image, threshold_point, pose):
        # print("goal_cat_id: ", goal_cat_id)
        # print("sem: ", sem.shape)
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
            local_ob_map, selem) != True
        # traversible = 1 - traversible
        planner = FMMPlanner(traversible)
        goal_pose_map = np.zeros((local_ob_map.shape))
        pose_x = int(pose[0]) if int(pose[0]) < local_w-1 else local_w-1
        pose_y = int(pose[1]) if int(pose[1]) < local_w-1 else local_w-1
        goal_pose_map[pose_x, pose_y] = 1
        # goal_map = skimage.morphology.binary_dilation(
        #     goal_pose_map, selem) != True
        # goal_map = 1 - goal_map
        planner.set_multi_goal(goal_pose_map)

        img_label, num = measure.label(image, connectivity=2, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        # print("img_label.dtype: ", img_label.dtype) # 480*480
        Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        Goal_point = np.zeros(img_label.shape)
        Goal_score = []

        dict_cost = {}
        for i in range(1, len(props)):
            # print("area: ", props[i].area)
            # dist = pu.get_l2_distance(props[i].centroid[0], pose[0], props[i].centroid[1], pose[1])
            dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
            dist_s = 8 if dist < 300 else 0
            
            cost = props[i].area + dist_s

            if props[i].area > threshold_point:
                dict_cost[i] = cost
        
        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)
            
            # print(dict_cost)
            for i, (key, value) in enumerate(dict_cost):
                # print(i, key)
                Goal_edge[img_label == key + 1] = 1
                Goal_point[int(props[key].centroid[0]), int(props[key].centroid[1])] = i+1 #
                Goal_score.append(value)
                if i == 3:
                    break

        return Goal_edge, Goal_point, Goal_score

    
    def configure_lm():
        """
        Configure the language model, tokenizer, and embedding generator function.

        Sets self.lm, self.lm_model, self.tokenizer, and self.embedder based on the
        selected language model inputted to this function.

        Args:
            lm: str representing name of LM to use

        Returns:
            None
        """
        lm_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

        lm_model.eval()
        lm_model = lm_model.to(device)

        def scoring_fxn(text):
            tokens_tensor = tokenizer.encode(text,
                                            add_special_tokens=False,
                                            return_tensors="pt").to(device)
            with torch.no_grad():
                output = lm_model(tokens_tensor, labels=tokens_tensor)
                loss = output[0]

                return -loss
        return scoring_fxn
    
    scoring_fxn = configure_lm()
    
    def draw_boxes_with_labels_matplotlib(image, boxes, scores, labels, save_path="glip_result.png"):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        for box, score, label in zip(boxes, scores, labels):

            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, f'{label}: {score:.2f}', color='white',
                    bbox=dict(facecolor='green', alpha=0.5))

        plt.axis("off")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            print(f"✅ Saved annotated image to: {save_path}")
        plt.show()


    def nms_filter(boxes, scores, labels, iou_threshold=0.5):
        """
        對重疊的 boxes 進行 NMS，只保留分數最高的那個
        :param boxes: numpy array, shape = (N, 4)
        :param scores: list of confidence scores
        :param labels: list of class names
        :return: filtered boxes, scores, labels
        """
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores)
        keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

        filtered_boxes = boxes_tensor[keep].numpy()
        filtered_scores = [scores[i] for i in keep]
        filtered_labels = [labels[i] for i in keep]

        return filtered_boxes, filtered_scores, filtered_labels

    def object_d(patch_pil, predictor, hm3d_category):
            """
            使用 GLIP 模型來檢測圖像中的物件，只返回 hm3d_category 中定義的物件（支援模糊比對）

            :param patch_pil: PIL 圖像
            :param predictor: 初始化的 GLIPPredictor 實例
            :param hm3d_category: 目標物件類別列表（應全為小寫）
            :return: 檢測到的物件名稱（過濾後、去重）
            """
            patch_np = np.array(patch_pil)  # 先轉為 NumPy 陣列

            # 如果 patch_pil 是 RGB 格式，轉成 OpenCV 使用的 BGR 格式
            patch_rgb = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)

            # 更自然語意的 prompt，提升推理效果
            prompt = '. '.join(hm3d_category) +'.'
            # prompt = f"A photo showing the following objects: {', '.join(hm3d_category)}."

            # 執行 GLIP 預測
            boxes, scores, labels = predictor.predict(patch_rgb, prompt)
            boxes, scores, labels = nms_filter(boxes, scores, labels, iou_threshold=0.5)
            # draw_boxes_with_labels_matplotlib(patch_np, boxes, scores, labels)

            # Image.fromarray(patch_rgb).save("vis_debug.png")
            # patch_pil.save("vis_debug.png")

            # 移除重複
            unique_objects = list(dict.fromkeys(labels))

            if not unique_objects:
                print("⚠️ No valid objects detected by GLIP.")
                return ["__none__"]

            print("Detected labels (fuzzy matched):", unique_objects)
            return unique_objects

    def get_caption(patch_pil,client):
        
        buffer = io.BytesIO()
        patch_pil.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        question = (
            "Describe only what is clearly visible in the image in 2–3 short sentences. "
            "Mention the room type if clear, and also describe the layout: is the space open, closed, or connected to other areas (e.g., doorways or passages)? "
            "List the main visible objects. Avoid guessing or imagining anything outside the image."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You describe indoor scenes. Always include visible room types, layout clues (like doors or openings), and main objects. Never assume the whole space is enclosed unless clearly shown."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        # {"type": "image_url", "image_url": {"url": "https://9579-111-251-83-230.ngrok-free.app/images/node.png"}}
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()

    
    def convert_caption_to_structured(client, caption):
        prompt = (
            "Convert the following scene description into structured format:\n"
            "Room type: <e.g., hallway, bedroom, kitchen>. If more than one is visible, list them (e.g., 'hallway and living room'). If unclear, use 'Unknown'.\n"
            "Visible objects: <list 2–3 clearly visible objects>\n"
            "Layout: <brief layout, such as open, doorway to another room, has window, narrow, corner>\n"
            "Use only what’s described. Do not add new guesses.\n\n"
            f"Scene description: {caption}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def get_target_room_types_from_priors(target):
        """
        根據 ROOM_OBJECT_PRIORS 取得目標物件最常出現的房型清單。
        
        參數:
            target (str): 目標物件名稱，例如 "chair"
            room_object_priors (dict): 房型對物件集合
            
        回傳:
            List[str]: 房型清單
        """
        ROOM_OBJECT_PRIORS = {
            "bedroom": {"bed"},
            "living room": {"sofa", "tv_monitor", "plant"},
            "lounge": {"sofa", "tv_monitor", "plant"},
            "dining room": {"chair"},
            "study": {"chair"},
            "office": {"chair", "plant"},
            "kitchen": {"chair"},
            "bathroom": {"toilet"},
            "restroom": {"toilet"},
            "balcony": {"plant"},
        }

        target = target.lower()
        rooms = []
        for room, objects in ROOM_OBJECT_PRIORS.items():
            # 注意這裡要先把物件轉小寫
            objects_lower = {o.lower() for o in objects}
            if target in objects_lower:
                rooms.append(room)
        return rooms
    

    def estimate_probabilities_with_cot(
        structured_descriptions: list,
        target: str,
        client,
        target_room_types: list = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        """
        根據多個structured描述，使用Chain-of-Thought一次估算所有分數。

        參數:
            structured_descriptions (list): 每個節點的structured描述 (str)。
            target (str): 目標物件名稱。
            target_room_types (list): 先驗房型列表（可選）。
            client: OpenAI API客戶端。
            model (str): 使用的模型。
            temperature (float): 生成溫度。

        回傳:
            List[dict]: 每個dict包含 'area', 'probability', 'reason'。
        """

        # 準備先驗房型描述
        if target_room_types and len(target_room_types) > 0:
            prior_text = (
                "According to prior knowledge, this object is commonly found in the following room types:\n"
                + ", ".join(target_room_types) + ".\n"
            )
        else:
            prior_text = ""

        prompt_intro = f"""
    You are an expert in visual scene understanding and navigation planning.

    {prior_text}

    For each structured description below, estimate the probability (0–1) that this area is promising to explore to find the specified target object '{target}'.
    This includes:
    - whether the target is visible in the current area, and
    - whether the area is *immediately adjacent* to spaces where the target is commonly found (for example, a doorway leading directly to a typical room type).
    Do not consider spaces that are only indirectly connected or general proximity.


    Please think step by step before giving the final probability.

    Follow this reasoning format for each area:
    Step 1: Is the target object explicitly mentioned in visible objects?

    Step 2: Is the target object commonly associated with the room type? If unknown, say so.

    Step 3: Does the layout or connectivity (e.g., doorway to another room) suggest potential access to areas where the target is likely to be found?

    Step 4: Combine evidence to estimate the probability (0--1) and explain the reasoning.

    Example Answer:
    [
    {{"area": 1, "probability": 0.5, "reason": "The target is not visible, but the area is a hallway connecting to bedrooms."}},
    {{"area": 2, "probability": 0.7, "reason": "The target is likely in this room type."}},
    {{"area": 3, "probability": 0.4, "reason": "The living room often connects to bedrooms even though the target is not visible."}}
    ]

    Respond ONLY with a JSON list in this format:
    [
    {{"area": 1, "probability": 0.5, "reason": "..."}},
    {{"area": 2, "probability": 0.7, "reason": "..."}}
    ]
    """

        # 組裝所有描述
        structured_text = ""
        for idx, s in enumerate(structured_descriptions):
            structured_text += f"\nArea {idx+1}:\n{s}\n"

        # 組合完整prompt
        full_prompt = prompt_intro + structured_text

        # 呼叫API
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You analyze visual scenes and provide probabilities based on step-by-step reasoning."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        # 先抓內容
        content = response.choices[0].message.content.strip()

        # 嘗試剝掉 ```json``` 或其他多餘標記
        if content.startswith("```"):
            content = re.sub(r"```[a-zA-Z]*", "", content)
            content = content.replace("```", "").strip()

        # 打印看一下
        print("========= LLM OUTPUT =========")
        print(content)
        print("========= END ================")

        # 嘗試解析
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print("[ERROR] JSON解析失敗，請檢查LLM輸出")
            raise e

        return result

    def score_from_structured_caption(caption,score):

        # 分行解析結構
        room_type = ""
        room_types = []
        visible_objects = []
        layout = ""

        # 分行解析結構
        lines = caption.lower().splitlines()
        for line in lines:
            if line.startswith("room type:"):
                room_type = line.replace("room type:", "").strip()
                room_types = [x.strip() for x in room_type.replace(",", " and ").split(" and ") if x.strip()]
            elif line.startswith("visible objects:"):
                visible_objects = [obj.strip() for obj in line.replace("visible objects:", "").split(",")]
            elif line.startswith("layout:"):
                layout = line.replace("layout:", "").strip()

        # Layout權重乘法
        layout_boost = 1.0
        if any(x in layout for x in ["open", "central", "doorway to another room", "doorway"]):
            layout_boost = 1.2
        elif any(x in layout for x in ["cramped", "dead-end", "corner"]):
            layout_boost = 0.9

        # Negative keyword進一步降低
        negative_keywords = ["stair", "step", "ladder"]
        if any(k in layout for k in negative_keywords):
            layout_boost *= 0.8
        if any(any(k in obj for k in negative_keywords) for obj in visible_objects):
            layout_boost *= 0.8

        score = min(score * layout_boost, 1.0)

        # 最後四捨五入
        return round(score, 2)

    
    def construct_dist(objects, temperature=2.0):

        if not objects or len(objects) == 0:
            dist = torch.ones(len(category_to_id)) / len(category_to_id)
            dist = dist * 0.2
            return dist
        else:
            query_str = "In the current view, there are "
            for ob in objects:
                query_str += ob + ", "
            query_str += "and"

            TEMP = []
            for label in category_to_id:
                TEMP_STR = query_str + " "
                TEMP_STR += label + "."

                score = scoring_fxn(TEMP_STR)
                TEMP.append(score)
            dist = torch.tensor(TEMP)

            return dist


    # Step 6. 完整的整合函數
    def build_topological_graph(e,obstacle_map, explored_map, target_edge, local_pose,
                                GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton, debug=False):

        H, W = obstacle_map.shape
        res = 5  # 每單位代表 5 公分
        agent_pos = (
            int(np.clip(local_pose[1].item() * 100 / res, 0, H - 1)),
            int(np.clip(local_pose[0].item() * 100 / res, 0, W - 1))
        )
        occupancy = builder.build_occupancy_map(obstacle_map, explored_map)
        skeleton = builder.build_skeleton(obstacle_map, explored_map)
        G = builder.skeleton_to_graph(skeleton)
        if G.number_of_nodes() == 0:
            print("⚠️ skeleton_to_graph()產生空圖")
            empty_G = nx.Graph()
            empty_G.add_node((0,0), type="agent")  # 隨便放個空節點避免None
            return empty_G
        # G = builder.connect_components(G)
        G = nx.minimum_spanning_tree(G)
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            largest = max(components, key=len)
            G = G.subgraph(largest).copy()

        G_stg = builder.extract_stg(G)
        G_stg = builder.remove_nodes_near_obstacles(G_stg, obstacle_map, min_distance=5)
        G_stg = builder.merge_close_junctions(G_stg, occupancy)
        G_stg = builder.filter_stg_by_short_edges(G_stg)
        G_stg, agent_node = builder.mark_agent_node(G_stg,agent_pos)
        if not nx.is_connected(G):
            components = list(nx.connected_components(G_stg))
            largest = max(components, key=len)
            G_stg = G_stg.subgraph(largest).copy()

        if G_stg is None or G_stg.number_of_nodes() == 0:
            print("⚠️ G_stg is empty! Returning dummy node.")
            empty_G = nx.Graph()
            empty_G.add_node((0,0), type="agent")
            return empty_G
        # G_stg = nx.minimum_spanning_tree(G_stg)
        # 視覺化
        if debug:
            # count = {'agent': 0, 'neighbor': 0, 'exploratory': 0, 'ordinary': 0}
            # for _, d in G.nodes(data=True):
            #     count[d['type']] += 1
            # print(f"📊 Node Distribution: {count}")

            try:
                builder.visualize_stg(occupancy,skeleton, G_stg)
            except Exception as e:
                print("🖼️ Plot error:", e)

        return G_stg


    def rotate_and_update_semmap(i,envs, goal_maps,found_goal, global_goals, sem_map_module, local_map, local_map_stair, local_pose, planner_pose_inputs,
                             target_edge_map, args, local_w, local_h, num_scenes, device, wait_env, finished,
                                GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton):
            # print("rotate_and_update_semmap")

            for e in range(num_scenes):
                obstacle_map = local_map[e, 0].cpu().numpy()
                explored_map = local_map[e, 1].cpu().numpy()
                target_edge = target_edge_map[e]
                pose = local_pose[e].cpu().numpy()
                if i % 3 == 2:
                    G_topo_list[e] = build_topological_graph(e,obstacle_map,explored_map,target_edge, pose,
                                    GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton)

            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = goal_maps[e]  # global_goals[e]
                p_input['map_target'] = target_point_map[e]  # global_goals[e]
                p_input['new_goal'] = 1
                p_input['found_goal'] = found_goal[e]
                # print("found_goal: ", found_goal[e])
                p_input["rotate_in_place"] = True
                p_input['wait'] = False
                planner_inputs[e]['graph'] = G_topo_list[e]
                if args.visualize or args.print_images:
                    p_input['map_edge'] = target_edge_map[e]
                    local_map[e, -1, :, :] = 1e-5
                    p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()

            obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

            # for e, x in enumerate(done):
                # print("done_r: ", x)

            return obs, _, done, infos
    def down_and_update_semmap(i,envs,goal_maps,found_goal, sem_map_module, local_map, local_map_stair, local_pose, planner_pose_inputs,
                             target_edge_map, args, local_w, local_h, num_scenes, device, wait_env, finished,
                                GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton):
            # print("down_and_update_semmap")

            for e in range(num_scenes):
                obstacle_map = local_map[e, 0].cpu().numpy()
                explored_map = local_map[e, 1].cpu().numpy()
                target_edge = target_edge_map[e]
                pose = local_pose[e].cpu().numpy()
                if i % 3 == 2:
                    G_topo_list[e] = build_topological_graph(e,obstacle_map,explored_map,target_edge, pose,
                                GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton)

            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = goal_maps[e]  # global_goals[e]
                p_input['map_target'] = target_point_map[e]  # global_goals[e]
                p_input['new_goal'] = 1
                p_input['found_goal'] = found_goal[e]
                p_input["look_down"] = True
                p_input['wait'] = False
                planner_inputs[e]['graph'] = G_topo_list[e]
                if args.visualize or args.print_images:
                    p_input['map_edge'] = target_edge_map[e]
                    local_map[e, -1, :, :] = 1e-5
                    p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()

            obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)



            return obs, _, done, infos
    
    def up_and_update_semmap(i,envs,goal_maps,found_goal, sem_map_module, local_map, local_map_stair, local_pose, planner_pose_inputs,
                             target_edge_map, args, local_w, local_h, num_scenes, device, wait_env, finished,
                                GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton):
            # print("up_and_update_semmap")

            for e in range(num_scenes):
                obstacle_map = local_map[e, 0].cpu().numpy()
                explored_map = local_map[e, 1].cpu().numpy()
                target_edge = target_edge_map[e]
                pose = local_pose[e].cpu().numpy()
                if i % 3 == 2:
                    G_topo_list[e] = build_topological_graph(e,obstacle_map,explored_map,target_edge, pose,
                                GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton)

            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = goal_maps[e]  # global_goals[e]
                p_input['map_target'] = target_point_map[e]  # global_goals[e]
                p_input['new_goal'] = 1
                p_input['found_goal'] = found_goal[e]
                p_input["look_up"] = True
                p_input['wait'] = False
                planner_inputs[e]['graph'] = G_topo_list[e]
                if args.visualize or args.print_images:
                    p_input['map_edge'] = target_edge_map[e]
                    local_map[e, -1, :, :] = 1e-5
                    p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()

            obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)


            return obs, _, done, infos

    def quantize_node(n, grid_size=10):
        y, x = n
        return (y // grid_size, x // grid_size)
    
    def local_to_full(node_local, lmb):
        """
        將local map中的節點座標轉成full map的座標

        參數:
            node_local: tuple (y, x)，在local map的座標
            lmb: list [row_start, row_end, col_start, col_end]，local map的邊界在full map的位置

        回傳:
            node_full: tuple (y, x)，在full map的座標
        """
        local_y, local_x = node_local
        full_y = lmb[0] + local_y
        full_x = lmb[2] + local_x
        return (full_y, full_x)
    
    def shift_graph(G, dy, dx):
        G_shifted = nx.Graph()
        for node, attr in G.nodes(data=True):
            y, x = node
            new_node = (y - dy, x - dx)
            G_shifted.add_node(new_node, **attr)
        for u, v, edge_attr in G.edges(data=True):
            u_new = (u[0] - dy, u[1] - dx)
            v_new = (v[0] - dy, v[1] - dx)
            G_shifted.add_edge(u_new, v_new, **edge_attr)
        return G_shifted
    
    def collect_subtree_nodes(G, start_node, parent_node):
        """
        從 start_node 出發，沿所有無向邊收集所有 reachable node
        不會回到 parent_node
        """
        visited = set()
        stack = [start_node]
        subtree_nodes = []

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            subtree_nodes.append(current)

            neighbors = [n for n in G.neighbors(current) if n != parent_node and n not in visited]
            stack.extend(neighbors)
        return subtree_nodes

    
    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    predictor = GLIPPredictor(config_file, weight_file)

    ### LLM 
    neighbor_targets = [None for _ in range(num_scenes)]  # 紀錄當前目標鄰居節點
    global_map = None
    global_skeleton = None
    GLOBAL_PREV_EXPLORED = None
    skeleton_generated_mask = None
   

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    eve_angle = np.asarray(
        [infos[env_idx]['eve_angle'] for env_idx
            in range(num_scenes)])
    

    increase_local_map, local_map, local_map_stair, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose, eve_angle)

    local_map[:, 0, :, :][local_map[:, 13, :, :] > 0] = 0


    actions = torch.randn(num_scenes, 2)*6
    # print("actions: ", actions.shape)
    cpu_actions = nn.Sigmoid()(actions).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]
    global_goals = [[min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                    for x, y in global_goals]

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
    goal_r = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
    


    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
        obstacle_map = local_map[e, 0].cpu().numpy()
        explored_map = local_map[e, 1].cpu().numpy()
        target_edge = target_edge_map[e]
        pose = local_pose[e].cpu().numpy()
        G_topo_list[e] = build_topological_graph(e,obstacle_map,explored_map,target_edge, pose,
                                GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton)

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]  # global_goals[e]
        p_input['map_target'] = target_point_map[e]  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = wait_env[e] or finished[e]
        planner_inputs[e]['graph'] = G_topo_list[e]
        if args.visualize or args.print_images:
            p_input['map_edge'] = target_edge_map[e]
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()

    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
    rgb_np = None

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    found = [0 for _ in range(num_scenes)]
    found_goal = [0 for _ in range(num_scenes)]
    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break
        # print("step: ", step)
        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps 
        # print("g_step: ", g_step)
        # print("l_step: ", l_step)

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            # print("done: ", x)
            # print(f"Agent {e} clear_flag from infos: {infos[e].get('clear_flag', 'N/A')}")
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
   
                wait_env[e] = 1.
                visited_nodes_full = set()
                G_topo_list = [None for _ in range(num_scenes)]
                init_map_and_pose_for_env(e)
                found = [0 for _ in range(num_scenes)]
                global_map = None
                global_skeleton = None
                GLOBAL_PREV_EXPLORED = None
                skeleton_generated_mask = None
                ini_r = False
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module

        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        eve_angle = np.asarray(
            [infos[env_idx]['eve_angle'] for env_idx
             in range(num_scenes)])
        # print("eve_angle_1: ", eve_angle)
        

        increase_local_map, local_map, local_map_stair, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose, eve_angle)
            
              
        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            loc_r = np.clip(loc_r, 0, local_w - 1)
            loc_c = np.clip(loc_c, 0, local_h - 1)
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

            # work for stairs in val
            # ------------------------------------------------------------------
            if args.eval:
            # # clear the obstacle during the stairs
                if loc_r > local_w: loc_r = local_w-1
                if loc_c > local_h: loc_c = local_h-1
                if infos[e]['clear_flag'] or local_map[e, 18, loc_r, loc_c] > 0.5:
                    stair_flag[e] = 1

                if stair_flag[e]:
                    # must > 0
                    if torch.any(local_map[e, 18, :, :] > 0.5):
                        local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]
                    local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]
            # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        
        # print("found_goal: ", found_goal[e])
        # Step 0：原地旋轉一圈（平視）觀察環境
        if l_step == 0 and not found_goal[e]:
            for e in range(num_scenes):

                step_masks[e]+=1

                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.

                            
                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                                local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                            int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                            (local_w, local_h),
                                                            (full_w, full_h))
                dy = lmb[e][0] - old_lmb[e][0]
                dx = lmb[e][2] - old_lmb[e][2]
                if G_topo_list[e] is None:
                    G_topo_list[e] = nx.Graph()
                else:
                    G_topo_list[e] = shift_graph(G_topo_list[e], dy, dx)

                old_lmb[e] = lmb[e].copy()

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                        lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

                if infos[e]['clear_flag']:
                    clear_flag[e] = 1

                if clear_flag[e]:
                    local_map[e].fill_(0.)
                    clear_flag[e] = 0

            # ------------------------------------------------------------------
                    
            ### select the frontier edge            
            # ------------------------------------------------------------------
            # Edge Update
            for e in range(num_scenes):

                ############################ choose global goal map #############################
                # choose global goal map
                _local_ob_map = local_map[e][0].cpu().numpy()
                local_ob_map[e] = cv2.dilate(_local_ob_map, kernel)

                show_ex = cv2.inRange(local_map[e][1].cpu().numpy(),0.1,1)
                    
                kernel = np.ones((5, 5), dtype=np.uint8)
                free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

                contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if len(contours)>0:
                    contour = max(contours, key = cv2.contourArea)
                    cv2.drawContours(local_ex_map[e],contour,-1,1,1)

                    # clear the boundary
                    local_ex_map[e, 0:2, 0:local_w]=0.0
                    local_ex_map[e, local_w-2:local_w, 0:local_w-1]=0.0
                    local_ex_map[e, 0:local_w, 0:2]=0.0
                    local_ex_map[e, 0:local_w, local_w-2:local_w]=0.0
                            
                    target_edge = np.zeros((local_w, local_h))
                    target_edge = local_ex_map[e]-local_ob_map[e]

                    target_edge[target_edge>0.8]=1.0
                    target_edge[target_edge!=1.0]=0.0

                    local_pose_map = [local_pose[e][1]*100/args.map_resolution, local_pose[e][0]*100/args.map_resolution]
                    target_edge_map[e], target_point_map[e], Goal_score = remove_small_points(_local_ob_map, target_edge, 4, local_pose_map) 
            


                    local_ob_map[e]=np.zeros((local_w,
                                    local_h))
                    local_ex_map[e]=np.zeros((local_w,
                                    local_h))
                        
                explored_map = local_map[e, 1].cpu().numpy()
                obstacle_map = local_map[e, 0].cpu().numpy()
                # if (is_surroundings_explored(explored_map, local_pose,e) or is_near_obstacle(obstacle_map, explored_map, local_pose, e)) and ini_r:
                # if is_surroundings_explored(explored_map, local_pose,e) and ini_r:
                #     print(f"🔄 Skip rotate: Agent {e} 周圍已全部探索")
                #     if g_step != 0:
                #         goal_maps = goal_r
                # else:
                for i in range(12):
                        ini_r = True
                        if g_step != 0:
                            goal_maps = goal_r

                            for e in range(num_scenes):
                                if found_goal[e]:
                                    found[e] = 1

                        obs, _, done, infos = rotate_and_update_semmap(i,
                                envs, goal_maps, found, global_goals, sem_map_module, local_map, local_map_stair, local_pose, planner_pose_inputs,
                                target_edge_map, args, local_w, local_h, num_scenes, device, wait_env, finished,
                                            GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton
                        )
                            # rgb_from_obs = obs[0, 0:3]  # shape: [3, 120, 160]
                            # rgb_np = rgb_from_obs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            # plt.imshow(rgb_np)
                            # plt.axis('off')
                            # plt.show()
                        poses = torch.from_numpy(np.asarray(
                                [infos[env_idx]['sensor_pose'] for env_idx
                                in range(num_scenes)])
                            ).float().to(device)

                        eve_angle = np.asarray(
                                [infos[env_idx]['eve_angle'] for env_idx
                                in range(num_scenes)])
                            # print("eve_angle_2: ", eve_angle)
                            

                        increase_local_map, local_map, local_map_stair, local_pose = \
                                sem_map_module(obs, poses, local_map, local_pose, eve_angle)
                        
                        for e, x in enumerate(done):
                            # print("done: ", x)
                            # print(f"Agent {e} clear_flag from infos: {infos[e].get('clear_flag', 'N/A')}")
                            if x:
                                spl = infos[e]['spl']
                                success = infos[e]['success']
                                dist = infos[e]['distance_to_goal']
                                spl_per_category[infos[e]['goal_name']].append(spl)
                                success_per_category[infos[e]['goal_name']].append(success)
                                if args.eval:
                                    episode_success[e].append(success)
                                    episode_spl[e].append(spl)
                                    episode_dist[e].append(dist)
                                    if len(episode_success[e]) == num_episodes:
                                        finished[e] = 1
                
                                wait_env[e] = 1.
                                init_map_and_pose_for_env(e)
                                found = [0 for _ in range(num_scenes)]
                                global_map = None
                                global_skeleton = None
                                GLOBAL_PREV_EXPLORED = None
                                skeleton_generated_mask = None
                        #---------------------------------------------------------------
                        obs, _, done, infos = down_and_update_semmap(i,
                                envs,goal_maps,found, sem_map_module, local_map, local_map_stair, local_pose, planner_pose_inputs,
                                target_edge_map, args, local_w, local_h, num_scenes, device, wait_env, finished,
                                            GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton
                            )
                        poses = torch.from_numpy(np.asarray(
                                [infos[env_idx]['sensor_pose'] for env_idx
                                in range(num_scenes)])
                            ).float().to(device)

                        eve_angle = np.asarray(
                                [infos[env_idx]['eve_angle'] for env_idx
                                in range(num_scenes)])
                            # print("eve_angle_2: ", eve_angle)
                            # rgb_from_obs = obs[0, 0:3]  # shape: [3, 120, 160]
                            # rgb_np = rgb_from_obs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # shape: [120, 160, 3]
                            # plt.imshow(rgb_np)
                            # plt.axis('off')
                            # plt.show()
                            

                        increase_local_map, local_map, local_map_stair, local_pose = \
                                sem_map_module(obs, poses, local_map, local_pose, eve_angle)
                        
                        for e, x in enumerate(done):
                            # print("done: ", x)
                            # print(f"Agent {e} clear_flag from infos: {infos[e].get('clear_flag', 'N/A')}")
                            if x:
                                spl = infos[e]['spl']
                                success = infos[e]['success']
                                dist = infos[e]['distance_to_goal']
                                spl_per_category[infos[e]['goal_name']].append(spl)
                                success_per_category[infos[e]['goal_name']].append(success)
                                if args.eval:
                                    episode_success[e].append(success)
                                    episode_spl[e].append(spl)
                                    episode_dist[e].append(dist)
                                    if len(episode_success[e]) == num_episodes:
                                        finished[e] = 1
                
                                wait_env[e] = 1.
                                init_map_and_pose_for_env(e)
                                found = [0 for _ in range(num_scenes)]
                                global_map = None
                                global_skeleton = None
                                GLOBAL_PREV_EXPLORED = None
                                skeleton_generated_mask = None
                        #---------------------------------------------------------------
                        obs, _, done, infos = up_and_update_semmap(i,
                                envs,goal_maps,found, sem_map_module, local_map, local_map_stair, local_pose, planner_pose_inputs,
                                target_edge_map, args, local_w, local_h, num_scenes, device, wait_env, finished,
                                            GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton
                            )
                        poses = torch.from_numpy(np.asarray(
                                [infos[env_idx]['sensor_pose'] for env_idx
                                in range(num_scenes)])
                            ).float().to(device)

                        eve_angle = np.asarray(
                                [infos[env_idx]['eve_angle'] for env_idx
                                in range(num_scenes)])
                            # print("eve_angle_2: ", eve_angle)
                            

                        increase_local_map, local_map, local_map_stair, local_pose = \
                                sem_map_module(obs, poses, local_map, local_pose, eve_angle)
                        for e, x in enumerate(done):
                            # print("done: ", x)
                            # print(f"Agent {e} clear_flag from infos: {infos[e].get('clear_flag', 'N/A')}")
                            if x:
                                spl = infos[e]['spl']
                                success = infos[e]['success']
                                dist = infos[e]['distance_to_goal']
                                spl_per_category[infos[e]['goal_name']].append(spl)
                                success_per_category[infos[e]['goal_name']].append(success)
                                if args.eval:
                                    episode_success[e].append(success)
                                    episode_spl[e].append(spl)
                                    episode_dist[e].append(dist)
                                    if len(episode_success[e]) == num_episodes:
                                        finished[e] = 1
                
                                wait_env[e] = 1.
                                init_map_and_pose_for_env(e)
                                found = [0 for _ in range(num_scenes)]
                                global_map = None
                                global_skeleton = None
                                GLOBAL_PREV_EXPLORED = None
                                skeleton_generated_mask = None
                        #---------------------------------------------------------------
                            
                        locs = local_pose.cpu().numpy()
                        planner_pose_inputs[:, :3] = locs + origins
                        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
                        for e in range(num_scenes):
                            r, c = locs[e, 1], locs[e, 0]
                            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                                int(c * 100.0 / args.map_resolution)]
                            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
                
                #---------------------------------------------------------------
                for e in range(num_scenes):

                    step_masks[e]+=1

                    if wait_env[e] == 1:  # New episode
                        wait_env[e] = 0.

                                
                    full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                                    local_map[e]
                    full_pose[e] = local_pose[e] + \
                        torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                                int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                                (local_w, local_h),
                                                                (full_w, full_h))

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                            lmb[e][0] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :,
                                            lmb[e, 0]:lmb[e, 1],
                                            lmb[e, 2]:lmb[e, 3]]
                    local_pose[e] = full_pose[e] - \
                        torch.from_numpy(origins[e]).to(device).float()

                    if infos[e]['clear_flag']:
                        clear_flag[e] = 1

                    if clear_flag[e]:
                        local_map[e].fill_(0.)
                        clear_flag[e] = 0

                # ------------------------------------------------------------------
                        
                ### select the frontier edge            
                # ------------------------------------------------------------------
                # Edge Update
                for e in range(num_scenes):

                    ############################ choose global goal map #############################
                    # choose global goal map
                    _local_ob_map = local_map[e][0].cpu().numpy()
                    local_ob_map[e] = cv2.dilate(_local_ob_map, kernel)

                    show_ex = cv2.inRange(local_map[e][1].cpu().numpy(),0.1,1)
                        
                    kernel = np.ones((5, 5), dtype=np.uint8)
                    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

                    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                    if len(contours)>0:
                        contour = max(contours, key = cv2.contourArea)
                        cv2.drawContours(local_ex_map[e],contour,-1,1,1)

                        # clear the boundary
                        local_ex_map[e, 0:2, 0:local_w]=0.0
                        local_ex_map[e, local_w-2:local_w, 0:local_w-1]=0.0
                        local_ex_map[e, 0:local_w, 0:2]=0.0
                        local_ex_map[e, 0:local_w, local_w-2:local_w]=0.0
                                
                        target_edge = np.zeros((local_w, local_h))
                        target_edge = local_ex_map[e]-local_ob_map[e]

                        target_edge[target_edge>0.8]=1.0
                        target_edge[target_edge!=1.0]=0.0

                        local_pose_map = [local_pose[e][1]*100/args.map_resolution, local_pose[e][0]*100/args.map_resolution]
                        target_edge_map[e], target_point_map[e], Goal_score = remove_small_points(_local_ob_map, target_edge, 4, local_pose_map) 
                


                        local_ob_map[e]=np.zeros((local_w,
                                        local_h))
                        local_ex_map[e]=np.zeros((local_w,
                                        local_h))
                # ------------------------------------------------------------------
                # Global Policy
                
                ##### LLM global score
                # ------------------------------------------------------------------
                unique_labels = np.unique(target_point_map[e])
                unique_labels = unique_labels[unique_labels > 0]

                print("Unique labels:", unique_labels)

                # (2) 取得現有節點的座標
                existing_nodes = np.array([n for n in G_topo_list[e].nodes])
                # 如果 existing_nodes 形狀為 (N,2)，才能用cKDTree
                if len(existing_nodes) == 0:
                    raise ValueError("G_topo_list[e] 中沒有任何節點，無法連接")

                # 建 KDTree
                obs_map = local_map[e][0].cpu().numpy()

                # 建 KDTree
                tree = cKDTree(existing_nodes)

                for idx in unique_labels:
                    coords = np.argwhere(target_point_map[e] == idx)
                    centroid = coords.mean(axis=0)
                    y, x = centroid

                    print("Adding frontier node at:", (y, x))

                    # 找最近節點
                    dist, nearest_idx = tree.query([y, x])
                    nearest_node = tuple(existing_nodes[nearest_idx])

                    # 先檢查兩點之間有沒有障礙物
                    rr, cc = line_1(
                        int(round(y)),
                        int(round(x)),
                        int(round(nearest_node[0])),
                        int(round(nearest_node[1]))
                    )

                    obstructed = False
                    for r, c in zip(rr, cc):
                        if obs_map[r, c] > 0.5:  # obstacle
                            obstructed = True
                            break

                    if obstructed:
                        print("Skipping edge to", nearest_node, "because of obstacle.")
                        continue

                    # 加 node
                    G_topo_list[e].add_node(
                        (y, x),
                        type="frontier",
                        label=int(idx)
                    )

                    # 加 edge
                    G_topo_list[e].add_edge(
                        (y, x),
                        nearest_node,
                        weight=dist
                    )

                    # print("Nodes in G_topo_list[e]:")
                    # for node, data in G_topo_list[e].nodes(data=True):
                    #     print(node, data)


                # ------------------------------------------------------------------
                    
                # 找出 agent node
                agent_nodes = [n for n, d in G_topo_list[e].nodes(data=True) if d.get("type") == "agent"]
                if agent_nodes:
                    agent_node = agent_nodes[0]
                    # 先抓 junction
                    junction_neighbors = [n for n in G_topo_list[e].neighbors(agent_node) if G_topo_list[e].nodes[n].get("type") == "junction"]
                    frontier_neighbors = [n for n in G_topo_list[e].neighbors(agent_node) if G_topo_list[e].nodes[n].get("type") == "frontier"]
                    if len(junction_neighbors) == 1:
                        # junction只有一個，就把termination也加進來
                        termination_neighbors = [n for n in G_topo_list[e].neighbors(agent_node) if G_topo_list[e].nodes[n].get("type") == "termination"]
                        frontier_neighbors = [n for n in G_topo_list[e].neighbors(agent_node) if G_topo_list[e].nodes[n].get("type") == "frontier"]
                        neighbors = junction_neighbors + termination_neighbors + frontier_neighbors
                    else:
                        neighbors = junction_neighbors + frontier_neighbors

                for neighbor in neighbors:
                    subtree_nodes = collect_subtree_nodes(G_topo_list[e], neighbor, agent_node)
                    print(f"從 {neighbor} 收集到的 subtree 節點:")
                    for n in subtree_nodes:
                        print(f" - {n}: {G_topo_list[e].nodes[n].get('type')}")

                    has_frontier = any(
                        G_topo_list[e].nodes[n].get("type") == "frontier"
                        for n in subtree_nodes
                    )
                    print(f"===> 此分支是否有 frontier: {has_frontier}")



                cn = infos[e]['goal_cat_id'] + 4
                cname = infos[e]['goal_name'] 
                frontier_score_list[e] = []
                frontier_score_dict[e] = {}
                has_frontier_dict = {}
                rgb_list=[]
                print("neighbors:",len(neighbors))
                        
                for n in neighbors:

                    x, y = n[1], n[0]  # 注意: (col, row)
                        # print("x: ", x)
                        # print("y: ", y)

                    agent_x, agent_y, agent_theta = local_pose[e].cpu().numpy()
                    r, c = [int( agent_x * 100.0 / args.map_resolution),
                                                int(agent_y * 100.0 / args.map_resolution)]
                    # print("r: ", r)
                    # print("c: ", c)
                    # print("agent_theta: ", agent_theta)

                    objs_list = []
                        # for se_cn in range(args.num_sem_categories - 1):
                        #     if local_map[e][se_cn + 4, fmb[0]:fmb[1], fmb[2]:fmb[3]].sum() != 0.:
                        #         objs_list.append(hm3d_category[se_cn])
                        # 計算 agent → neighbor 向量的方向角
                    # 獲取面對該節點的 RGB 圖像
                    # 計算當前位置到目標節點的方向
                    # 轉成 pixel index
                    # 轉成弧度
                    if abs(agent_theta) > 2 * np.pi:
                        agent_theta = np.deg2rad(agent_theta)
                    else:
                        # 確保 agent_theta 在 [-π, π] 範圍內
                        agent_theta = (agent_theta + np.pi) % (2 * np.pi) - np.pi

                    # print("target_node (x, y):", x, y)
                    dx = x - r
                    dy = y - c
                    target_theta = np.arctan2(dy, dx)
                    # print("agent_theta (rad):", agent_theta)
                    # print("target_theta (rad):", target_theta)
                    
                    # 計算需要旋轉的角度
                    rotation_angle = target_theta - agent_theta
                    # 將角度規範化到 [-π, π] 範圍
                    rotation_angle = (rotation_angle + np.pi) % (2 * np.pi) - np.pi
                    # print("rotation_angle (rad):", rotation_angle)
                    
                    # 將角度轉換為旋轉步數，使用四捨五入而不是取整
                    rotation_steps = round(rotation_angle / (np.pi/6))
                    # print("rotation_steps:", rotation_steps)
                    
                    # 限制最大旋轉次數，避免無限旋轉
                    max_steps = 6  # 最多旋轉180度
                    rotation_steps = np.clip(rotation_steps, -max_steps, max_steps)
                    # print("rotation_steps: ", rotation_steps)
            
                    if rotation_steps != 0:
                        # 執行旋轉
                        for _ in range(abs(rotation_steps)):
                            if rotation_steps > 0:
                                # 順時針旋轉
                                planner_inputs = [{} for _ in range(num_scenes)]
                                for p_input in planner_inputs:
                                    p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                                    p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                                    p_input['pose_pred'] = planner_pose_inputs[e]
                                    p_input['goal'] = goal_maps[e]
                                    p_input['map_target'] = target_point_map[e]
                                    p_input['new_goal'] = 1
                                    p_input['found_goal'] = found_goal[e]
                                    p_input["rotate_in_place_l"] = True
                                    p_input['wait'] = False
                                    p_input['graph'] = G_topo_list[e]
                                    if args.visualize or args.print_images:
                                        p_input['map_edge'] = target_edge_map[e]
                                        local_map[e, -1, :, :] = 1e-5
                                        p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                                            ].argmax(0).cpu().numpy()
                                    
                                obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
                                poses = torch.from_numpy(np.asarray(
                                    [infos[env_idx]['sensor_pose'] for env_idx
                                    in range(num_scenes)])
                                ).float().to(device)

                                eve_angle = np.asarray(
                                        [infos[env_idx]['eve_angle'] for env_idx
                                        in range(num_scenes)])
                                increase_local_map, local_map, local_map_stair, local_pose = \
                                        sem_map_module(obs, poses, local_map, local_pose, eve_angle)
                                
                                for e, x in enumerate(done):
                                    # print("done: ", x)
                                    # print(f"Agent {e} clear_flag from infos: {infos[e].get('clear_flag', 'N/A')}")
                                    if x:
                                        spl = infos[e]['spl']
                                        success = infos[e]['success']
                                        dist = infos[e]['distance_to_goal']
                                        spl_per_category[infos[e]['goal_name']].append(spl)
                                        success_per_category[infos[e]['goal_name']].append(success)
                                        if args.eval:
                                            episode_success[e].append(success)
                                            episode_spl[e].append(spl)
                                            episode_dist[e].append(dist)
                                            if len(episode_success[e]) == num_episodes:
                                                finished[e] = 1
                        
                                        wait_env[e] = 1.
                                        init_map_and_pose_for_env(e)
                                        found = [0 for _ in range(num_scenes)]
                                        global_map = None
                                        global_skeleton = None
                                        GLOBAL_PREV_EXPLORED = None
                                        skeleton_generated_mask = None

                                locs = local_pose.cpu().numpy()
                                planner_pose_inputs[:, :3] = locs + origins
                                local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
                                for e in range(num_scenes):
                                    r, c = locs[e, 1], locs[e, 0]
                                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                                        int(c * 100.0 / args.map_resolution)]
                                    local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
                            else:
                                # 逆時針旋轉
                                planner_inputs = [{} for _ in range(num_scenes)]
                                for p_input in planner_inputs:
                                    p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                                    p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                                    p_input['pose_pred'] = planner_pose_inputs[e]
                                    p_input['goal'] = goal_maps[e]
                                    p_input['map_target'] = target_point_map[e]
                                    p_input['new_goal'] = 1
                                    p_input['found_goal'] = found_goal[e]
                                    p_input["rotate_in_place"] = True
                                    p_input['wait'] = False
                                    p_input['graph'] = G_topo_list[e]
                                    if args.visualize or args.print_images:
                                        p_input['map_edge'] = target_edge_map[e]
                                        local_map[e, -1, :, :] = 1e-5
                                        p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                                            ].argmax(0).cpu().numpy()
                                
                                obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
                                for e, x in enumerate(done):
                                    # print("done: ", x)
                                    # print(f"Agent {e} clear_flag from infos: {infos[e].get('clear_flag', 'N/A')}")
                                    if x:
                                        spl = infos[e]['spl']
                                        success = infos[e]['success']
                                        dist = infos[e]['distance_to_goal']
                                        spl_per_category[infos[e]['goal_name']].append(spl)
                                        success_per_category[infos[e]['goal_name']].append(success)
                                        if args.eval:
                                            episode_success[e].append(success)
                                            episode_spl[e].append(spl)
                                            episode_dist[e].append(dist)
                                            if len(episode_success[e]) == num_episodes:
                                                finished[e] = 1
                        
                                        wait_env[e] = 1.
                                        init_map_and_pose_for_env(e)
                                        found = [0 for _ in range(num_scenes)]
                                        global_map = None
                                        global_skeleton = None
                                        GLOBAL_PREV_EXPLORED = None
                                        skeleton_generated_mask = None

                                poses = torch.from_numpy(np.asarray(
                                        [infos[env_idx]['sensor_pose'] for env_idx
                                        in range(num_scenes)])
                                    ).float().to(device)

                                eve_angle = np.asarray(
                                        [infos[env_idx]['eve_angle'] for env_idx
                                        in range(num_scenes)])
                                increase_local_map, local_map, local_map_stair, local_pose = \
                                        sem_map_module(obs, poses, local_map, local_pose, eve_angle)

                        
                                locs = local_pose.cpu().numpy()
                                planner_pose_inputs[:, :3] = locs + origins
                                local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
                                for e in range(num_scenes):
                                    r, c = locs[e, 1], locs[e, 0]
                                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                                        int(c * 100.0 / args.map_resolution)]
                                    local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
                    else:
                        planner_inputs = [{} for _ in range(num_scenes)]
                        for p_input in planner_inputs:
                            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                            p_input['pose_pred'] = planner_pose_inputs[e]
                            p_input['goal'] = goal_maps[e]
                            p_input['map_target'] = target_point_map[e]
                            p_input['new_goal'] = 1
                            p_input['found_goal'] = 0
                            p_input["rotate_in_place"] = True
                            p_input['wait'] = False
                            p_input['graph'] = G_topo_list[e]
                            if args.visualize or args.print_images:
                                p_input['map_edge'] = target_edge_map[e]
                                local_map[e, -1, :, :] = 1e-5
                                p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                                            ].argmax(0).cpu().numpy()
                                
                            obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
                            for e, x in enumerate(done):
                                    # print("done: ", x)
                                    # print(f"Agent {e} clear_flag from infos: {infos[e].get('clear_flag', 'N/A')}")
                                    if x:
                                        spl = infos[e]['spl']
                                        success = infos[e]['success']
                                        dist = infos[e]['distance_to_goal']
                                        spl_per_category[infos[e]['goal_name']].append(spl)
                                        success_per_category[infos[e]['goal_name']].append(success)
                                        if args.eval:
                                            episode_success[e].append(success)
                                            episode_spl[e].append(spl)
                                            episode_dist[e].append(dist)
                                            if len(episode_success[e]) == num_episodes:
                                                finished[e] = 1
                        
                                        wait_env[e] = 1.
                                        init_map_and_pose_for_env(e)
                                        found = [0 for _ in range(num_scenes)]
                                        global_map = None
                                        global_skeleton = None
                                        GLOBAL_PREV_EXPLORED = None
                                        skeleton_generated_mask = None

                            poses = torch.from_numpy(np.asarray(
                                        [infos[env_idx]['sensor_pose'] for env_idx
                                        in range(num_scenes)])
                                ).float().to(device)

                            eve_angle = np.asarray(
                                        [infos[env_idx]['eve_angle'] for env_idx
                                        in range(num_scenes)])
                            increase_local_map, local_map, local_map_stair, local_pose = \
                                        sem_map_module(obs, poses, local_map, local_pose, eve_angle)

                        
                            locs = local_pose.cpu().numpy()
                            planner_pose_inputs[:, :3] = locs + origins
                            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
                            for e in range(num_scenes):
                                r, c = locs[e, 1], locs[e, 0]
                                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                                        int(c * 100.0 / args.map_resolution)]
                                local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.    
                    # 轉換為 PIL 圖像
                    img_pil = Image.open("images/node.png").convert("RGB").copy()
                    rgb_list.append(img_pil)

                    # plt.imshow(rgb_np)
                    # plt.axis('off')
                    # plt.show()
                
                # 一次批量跑 object detection
                batch_objs_list = []
                for rgb_np in rgb_list:
                    objs_list = object_d(rgb_np, predictor, GLIP_category)
                    batch_objs_list.append(objs_list)

                # 一次批量跑 caption + structured
                batch_structured = []
                for rgb_np in rgb_list:
                    caption = get_caption(rgb_np, client)
                    print("[Caption]:", caption)
                    structured = convert_caption_to_structured(client, caption)
                    print("[Structured]:", structured)
                    batch_structured.append(structured)

                target_room_types = get_target_room_types_from_priors(cname)
                discoverability_results = estimate_probabilities_with_cot(batch_structured,cname,client,target_room_types)

                # 處理分數
                for idx, n in enumerate(neighbors):
                    objs_list = batch_objs_list[idx]
                    structured = batch_structured[idx]

                    # object detection分數
                    ref_dist = F.softmax(construct_dist(objs_list) / 2, dim=0).to(device)
                    new_dist = ref_dist
                    score = new_dist[category_to_id.index(cname)]
                    room_score = discoverability_results[idx]['probability']
                    reason = discoverability_results[idx]['reason']

                    room_score = score_from_structured_caption(structured,room_score)

                    print(f"節點 {n}: 探索分數 = {room_score:.2f}, 理由 = {reason}")
                    print(f"Agent {e} 在節點 {n} 的分數: {score:.3f}, 房間分數: {room_score:.3f}")
                    region = quantize_node(n, grid_size=20)
                    # penalty = 0.7 ** region_visit_counter[e][region]
                    # parse room types (複製 score_from_structured_caption 裡的一段)
                    lines = structured.lower().splitlines()
                    room_types = []
                    for line in lines:
                        if line.startswith("room type:"):
                            room_type = line.replace("room type:", "").strip()
                            room_types = [x.strip() for x in room_type.replace(",", " and ").split(" and ") if x.strip()]

                    penalty_base = 0.9
                    is_hallway = False
                    for rt in room_types:
                        if "hallway" in rt or "corridor" in rt:
                            is_hallway = True
                            break
                    if not is_hallway:
                        penalty_base = 0.6
                        print("not hallway")

                    penalty = penalty_base ** region_visit_counter[e][region]

                    frontier_score_dict[e][n] = (0.3 * score + 0.7 * room_score) * penalty
                    print(f"Agent {e} 在節點 {n} 的最終分數:{frontier_score_dict[e][n]}")

                        
                print("frontier_score_dict", len(frontier_score_dict[e]))        

            if frontier_score_dict[e]:
                # 初始化 traversible map（反障礙物區）
                local_ob = local_map[e, 0].cpu().numpy()
                traversible = skimage.morphology.binary_dilation(local_ob, skimage.morphology.disk(1)) != True
                planner = FMMPlanner(traversible)

                # 取得 agent 目前位置
                r, c, _ = local_pose[e].cpu().numpy()
                start_r = int(r * 100 / args.map_resolution)
                start_c = int(c * 100 / args.map_resolution)
                start_r = np.clip(start_r, 0, local_w - 1)
                start_c = np.clip(start_c, 0, local_h - 1)

                start_map = np.zeros((local_w, local_h))
                start_map[start_r, start_c] = 1
                planner.set_multi_goal(start_map)
                

                frontier_nodes = {}
                non_frontier_nodes = {}

                for node, score in frontier_score_dict[e].items():
                    if has_frontier_dict.get(node, False):
                        frontier_nodes[node] = score
                    else:
                        non_frontier_nodes[node] = score

                # 優先使用有frontier的節點
                selected_scores = frontier_nodes if frontier_nodes else non_frontier_nodes

                if selected_scores:
                    best_neighbor = max(selected_scores, key=selected_scores.get)
                    best_score = selected_scores[best_neighbor]
                    neighbor_targets[e] = best_neighbor

                    region = quantize_node(best_neighbor)
                    region_visit_counter[e][region] += 1

                    # print("Local coord:", best_neighbor)
                    # print("lmb:", lmb[e])
                    # print("Full coord:", local_to_full(best_neighbor, lmb[e]))

                    best_neighbor_full = local_to_full(best_neighbor, lmb[e])
                    q_node = quantize_node(best_neighbor_full, grid_size=5)
                    visited_nodes_full.add(q_node)
                    region_visit_counter[e][q_node] += 1

                    if q_node in visited_nodes_full:
                        print("✅ 這個區域已經走過")

                    print(f"✅ Agent {e} 選擇節點 {best_neighbor} (分數 = {best_score:.3f})")
                else:
                    print(f"⚠️ Agent {e} 沒有可達的節點，將使用 fallback")
                    neighbor_targets[e] = None
        # ------------------------------------------------------------------
            ##### select randomly point
            # ------------------------------------------------------------------
            actions = torch.randn(num_scenes, 2)*6
            cpu_actions = nn.Sigmoid()(actions).numpy()
            global_goals = [[int(action[0] * local_w),
                                int(action[1] * local_h)]
                            for action in cpu_actions]
            global_goals = [[min(x, int(local_w - 1)),
                                min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_masks = torch.ones(num_scenes).float().to(device)

            # --------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
    
        local_goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]


        for e in range(num_scenes):

            # ------------------------------------------------------------------
            if neighbor_targets[e] is not None:
                goal_y, goal_x = neighbor_targets[e]  # 注意：G_topo 節點為 (y, x)
                gy = int(round(goal_y))
                gx = int(round(goal_x))
                gy = np.clip(gy, 0, local_w - 1)
                gx = np.clip(gx, 0, local_h - 1)
                local_goal_maps[e][gy, gx] = 1
                g_sum_global += 1
            else:
                # fallback: 使用隨機點（保險做法）
                local_goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
                # print("Don't Find the edge")

            cn = infos[e]['goal_cat_id'] + 4
            if local_map[e, cn, :, :].sum() != 0.:
                # print("Find the target")
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                if cn == 9:
                    cat_semantic_scores = cv2.dilate(cat_semantic_scores, tv_kernel)
                local_goal_maps[e] = find_big_connect(cat_semantic_scores)
                found_goal[e] = 1

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            obstacle_map = local_map[e, 0].cpu().numpy()
            explored_map = local_map[e, 1].cpu().numpy()
            target_edge = target_edge_map[e]
            pose = local_pose[e].cpu().numpy()
            if l_step % 5 == 4:
                G_topo_list[e] = build_topological_graph(e,obstacle_map,explored_map,target_edge, pose,
                                    GLOBAL_PREV_EXPLORED, skeleton_generated_mask,global_map,global_skeleton)
            # planner_pose_inputs[e, 3:] = [0, local_w, 0, local_h]
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = local_goal_maps[e]  # global_goals[e]
            p_input['map_target'] = target_point_map[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = False
            planner_inputs[e]['graph'] = G_topo_list[e]
            if args.visualize or args.print_images:
                p_input['map_edge'] = target_edge_map[e]
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                :].argmax(0).cpu().numpy()
   

        obs, fail_case, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # rgb_from_obs = obs[0, 0:3]  # shape: [3, 120, 160]
        # rgb_np = rgb_from_obs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # shape: [120, 160, 3]
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            log += "\n\tLLM Rewards: " + str(g_process_rewards /g_sum_rewards)
            log += "\n\tLLM use rate: " + str(g_sum_rewards /g_sum_global)

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl))

                total_collision = []
                total_exploration = []
                total_detection = []
                total_success = []
                for e in range(args.num_processes):
                    total_collision.append(fail_case[e]['collision'])
                    total_exploration.append(fail_case[e]['exploration'])
                    total_detection.append(fail_case[e]['detection'])
                    total_success.append(fail_case[e]['success'])

                if len(total_spl) > 0:
                    log += " Fail Case: collision/exploration/detection/success:"
                    log += " {:.0f}/{:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
                        np.sum(total_collision),
                        np.sum(total_exploration),
                        np.sum(total_detection),
                        np.sum(total_success),
                        len(total_spl))


            print(log)
            logging.info(log)
        # ------------------------------------------------------------------


    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        log += "\n\tLLM Rewards: " + str(g_process_rewards /g_sum_rewards)
        log += "\n\tLLM use rate: " + str(g_sum_rewards /g_sum_global)


        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)



if __name__ == "__main__":
    main()
