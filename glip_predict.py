import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
import sys
sys.path.append("/home/ting/llm/GLIP")
from GLIP.maskrcnn_benchmark.config import cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

class GLIPPredictor:
    def __init__(self, config_file, weight_file, device="cuda", threshold=0.6, min_image_size=800):
        """
        初始化 GLIP 模型
        :param config_file: 配置文件路徑
        :param weight_file: 權重文件路徑
        :param device: 運行設備（cuda 或 cpu）
        :param threshold: 預測的置信度閾值
        :param min_image_size: 最小圖像尺寸
        """
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", device])
        cfg.merge_from_list(["MODEL.ROI_HEADS.SCORE_THRESH", threshold])

        self.glip_demo = GLIPDemo(
            cfg,
            min_image_size=min_image_size,
            confidence_threshold=threshold,
            show_mask_heatmaps=False
        )
        self.threshold = threshold

    def predict(self, image, caption):
        """
        進行 GLIP 物件偵測推理
        :param image: 輸入影像 (numpy array or PIL image)
        :param caption: 物件偵測描述文本
        :return: 檢測到的邊界框、分數、標籤名稱
        """
        preds = self.glip_demo.compute_prediction(image, caption)
        top_preds = self.glip_demo._post_process(preds, threshold=0.6)

        # 从预测结果中提取预测类别,得分和检测框
        labels = top_preds.get_field("labels").tolist()
        scores = top_preds.get_field("scores").tolist()
        boxes = top_preds.bbox.detach().cpu().numpy()

        # 获得标签数字对应的类别名
        labels_names = self.glip_demo.get_label_names(labels)

        return boxes, scores, labels_names
