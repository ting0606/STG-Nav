import cv2
import numpy as np


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image(goal_name, legend):
    vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations (Goal: {})".format(goal_name)
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color

    # draw legend
    #lx, ly, _ = legend.shape
    #vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image

def draw_legend():
    colors = [
        (247, 247, 247),  # 0: Navigable Area (白色)
        (170, 200, 228),  # 1: Chair (淡棕色)
        (172, 225, 235),  # 2: Couch (黃棕色)
        (174, 237, 229),  # 3: Potted Plant (淡綠色)
        (174, 237, 209),  # 4: Bed (綠色)
        (173, 234, 192),  # 5: Toilet (亮綠色)
        (194, 235, 188),  # 6: TV (藍綠色)
        (218, 235, 190),  # 7: Dining Table (淺藍色)
        (238, 232, 190),  # 8: Oven (青綠色)
        (236, 206, 179),  # 9: Sink (淺青色)
        (237, 187, 180),  # 10: Refrigerator (淡藍色)
        (234, 170, 176),  # 11: Book (紫色)
        (233, 170, 197),  # 12: Clock (藍紫色)
        (234, 171, 219),  # 13: Vase (粉紅色)
        (213, 171, 223),  # 14: Cup (淡粉色)
        (189, 171, 223)   # 15: Bottle (淡紅色)
    ]

    # **圖例標籤**
    labels = [
        "Navigable Area", "0: Chair", "1: Couch", "2: Potted Plant", "3: Bed", "4: Toilet",
        "5: TV", "6: Dining Table", "7: Oven", "8: Sink", "9: Refrigerator", "10: Book",
        "11: Clock", "12: Vase", "13: Cup", "14: Bottle"
    ]
    
    cols = 4  # 每行 4 個類別
    rows = int(np.ceil(len(labels) / cols))
    box_size = 20  # 顏色塊大小
    spacing_x = 220  # 水平間距
    spacing_y = 25  # 行間距（增加，使字體不擠）

    width = cols * spacing_x + 10
    height = rows * spacing_y + 5  # 調整高度，確保不壓到字

    legend = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        row = i // cols
        col = i % cols
        
        x = col * spacing_x + 30  # 調整間距
        y = row * spacing_y + 5  # 增加 Y 位置，讓字體有更多空間
        
        # 畫顏色方塊
        cv2.rectangle(legend, (x, y), (x + box_size, y + box_size), color, -1)
        # 畫標籤，稍微向右移動，避免顏色方塊擋住
        cv2.putText(legend, label, (x + box_size + 15, y + box_size - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return legend



