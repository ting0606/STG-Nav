
import numpy as np
from skimage.morphology import skeletonize, thin, binary_closing, remove_small_objects
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import glob
from skimage import measure

class STGBuilder:
    def __init__(self, erosion_size=2, exp_thresh=0.5, merge_dist_thresh=15, min_edge_length=10):
        self.erosion_size = erosion_size
        self.exp_thresh = exp_thresh
        self.merge_dist_thresh = merge_dist_thresh
        self.min_edge_length = min_edge_length

    def build_occupancy_map(self, obs, exp):
        occupancy_map = np.full_like(exp, -1, dtype=int)
        occupancy_map[(exp > 0.5) & (obs == 0)] = 0
        occupancy_map[(obs > 0.5)] = 100
        return occupancy_map

    def build_skeleton(self, obs, exp):
        raw_free = (exp > self.exp_thresh) & (obs <= 0.5)
        free_uint8 = raw_free.astype(np.uint8)
        kernel = np.ones((2 * self.erosion_size + 1, 2 * self.erosion_size + 1), np.uint8)
        eroded_free = cv2.erode(free_uint8, kernel, iterations=1)
        cleaned = binary_closing(eroded_free, selem=np.ones((3, 3)))
        cleaned = remove_small_objects(cleaned, min_size=5)
        skeleton = thin(cleaned).astype(np.uint8)
        # 不只保留最大區域，而是保留所有夠大的
        min_keep_area = 20
        labeled = label(skeleton, connectivity=2)
        regions = regionprops(labeled)
        skeleton_cleaned = np.zeros_like(skeleton)
        for region in regions:
            if region.area >= min_keep_area:
                skeleton_cleaned[labeled == region.label] = 1
        return skeleton_cleaned


    def skeleton_to_graph(self, skeleton_mask):
        coords = np.argwhere(skeleton_mask > 0)
        G = nx.Graph()
        for y, x in coords:
            neighbors = [(y+dy, x+dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not (dy == 0 and dx == 0)]
            for ny, nx_ in neighbors:
                if 0 <= ny < skeleton_mask.shape[0] and 0 <= nx_ < skeleton_mask.shape[1]:
                    if skeleton_mask[ny, nx_] > 0:
                        G.add_edge((y, x), (ny, nx_), pixels=[(y, x), (ny, nx_)])
        return G

    def extract_stg(self, G_mst):
        node_types = {}
        for node in G_mst.nodes:
            deg = G_mst.degree(node)
            if deg == 1:
                node_types[node] = "termination"
            elif deg >= 3:
                node_types[node] = "junction"
            else:
                node_types[node] = "path"

        # 2. 收集關鍵節點
        key_nodes = [n for n, t in node_types.items() if t in ("termination", "junction")]

        # 3. 初始化新Graph
        G_simplified = nx.Graph()
        for n in key_nodes:
            G_simplified.add_node(n, type=node_types[n])

        # 4. 遍歷每個關鍵節點，沿著path延伸
        visited = set()

        for start in key_nodes:
            neighbors = list(G_mst.neighbors(start))
            for nbr in neighbors:
                if (start, nbr) in visited or (nbr, start) in visited:
                    continue
                path = [start, nbr]
                current = nbr
                prev = start
                while node_types[current] == "path":
                    next_nodes = [n for n in G_mst.neighbors(current) if n != prev]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    path.append(next_node)
                    prev = current
                    current = next_node
                # current now is either termination or junction
                if node_types.get(current) in ("termination", "junction"):
                    G_simplified.add_node(current, type=node_types[current])
                    G_simplified.add_edge(start, current, pixels=path)
                    # Mark all edges visited
                    for i in range(len(path)-1):
                        visited.add((path[i], path[i+1]))

        return G_simplified
    
    # def merge_close_junctions(self, G_stg):
    #     junction_nodes = [n for n, d in G_stg.nodes(data=True) if d.get('type') == 'junction']
    #     if len(junction_nodes) < 2:
    #         return G_stg

    #     coords = np.array(junction_nodes)
    #     clustering = DBSCAN(eps=14, min_samples=1).fit(coords)
    #     labels = clustering.labels_

    #     merged_graph = G_stg.copy()
    #     for label in set(labels):
    #         group = coords[labels == label]
    #         if len(group) <= 1:
    #             continue
    #         # 新node取平均
    #         new_node = tuple(np.mean(group, axis=0).astype(int))
    #         merged_graph.add_node(new_node, type='junction')

    #         # 與group中每個node的鄰居重新連線
    #         for old_coord in group:
    #             old = tuple(old_coord)
    #             for neighbor in list(merged_graph.neighbors(old)):
    #                 if neighbor not in [tuple(g) for g in group]:
    #                     pixels = merged_graph.edges[old, neighbor].get('pixels', [old, neighbor])
    #                     merged_graph.add_edge(new_node, neighbor, pixels=pixels)
    #             merged_graph.remove_node(old)
    #     return merged_graph

    def merge_close_junctions(self, G_stg, occupancy_map, scaling_factor=0.05):
        """
        根據已探索區域 bounding box 計算 eps
        """
        junction_nodes = [n for n, d in G_stg.nodes(data=True) if d.get('type') == 'junction']
        if len(junction_nodes) < 2:
            return G_stg

        coords = np.array(junction_nodes)

        # 根據occupancy map自動計算eps
        if occupancy_map is not None:
            # 將已知區域mask (自由區域或障礙物)
            known_mask = (occupancy_map == 0) | (occupancy_map == 100)
            ys, xs = np.where(known_mask)
            if len(ys) >= 2:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                diag = np.hypot(y_max - y_min, x_max - x_min)
                eps = diag * scaling_factor

                if eps < 15:
                    eps = 14
            else:
                eps = 14  # fallback
        else:
            eps = 14

        print("[merge_close_junctions] eps =", eps)

        clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
        labels = clustering.labels_

        merged_graph = G_stg.copy()
        for label in set(labels):
            group = coords[labels == label]
            if len(group) <= 1:
                continue
            new_node = tuple(np.mean(group, axis=0).astype(int))
            merged_graph.add_node(new_node, type='junction')
            for old_coord in group:
                old = tuple(old_coord)
                for neighbor in list(merged_graph.neighbors(old)):
                    if neighbor not in [tuple(g) for g in group]:
                        pixels = merged_graph.edges[old, neighbor].get('pixels', [old, neighbor])
                        merged_graph.add_edge(new_node, neighbor, pixels=pixels)
                merged_graph.remove_node(old)
        return merged_graph

    def filter_stg_by_short_edges(self, G_stg):
        to_remove = []
        for node in list(G_stg.nodes):
            if G_stg.degree(node) == 1 and G_stg.nodes[node].get('type') == 'termination':
                neighbor = next(G_stg.neighbors(node))
                edge_data = G_stg.get_edge_data(node, neighbor)
                if edge_data and 'pixels' in edge_data and len(edge_data['pixels']) < self.min_edge_length:
                    to_remove.append(node)
        G_filtered = G_stg.copy()
        G_filtered.remove_nodes_from(to_remove)
        return G_filtered
    
    def extract_frontiers(self,obs, exp):
        """
        從 occupancy map 中找出 frontier 點
        frontier: 自由區域旁邊是未知區域的邊界點
        """
        local_ex_map = np.zeros_like(obs, dtype=np.uint8)  # dtype 改為 uint8
        local_ob_map = np.zeros_like(obs, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        local_ob_map = cv2.dilate(obs, kernel)
        frontier_mask = np.zeros_like(obs, dtype=np.float32)  # 如果你之後要處理 float 值

        show_ex = cv2.inRange(exp, 0.1, 1)  # 將 exp 範圍為 [0.1, 1.0] 的區域取出來
        kernel = np.ones((5, 5), dtype=np.uint8)
        free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(local_ex_map, [contour], -1, 1, 1)  # 注意 drawContours 第二個參數要用 list 包起來

        # 原本這裡 frontier_mask 被初始化為 0，再減掉 local_ob_map 是沒有意義的，應改成：
        frontier_mask = local_ex_map.astype(np.float32)  # 轉為 float 做後續處理
        frontier_mask[local_ob_map > 0] = 0  # 去除靠近障礙物的區域
        frontier_mask[frontier_mask > 0.8] = 1.0
        frontier_mask[frontier_mask != 1.0] = 0.0

        return frontier_mask

    def extract_frontier_centers(self, frontier_mask, fmm_planner, pose, min_area=10, min_dist=20, max_dist=500, max_centers=4):
        """
        將 frontier mask 的連通區域轉成最多 max_centers 個中心點，
        根據面積 + FMM距離排序
        """
        labeled, num = measure.label(frontier_mask, connectivity=2, return_num=True)
        props = measure.regionprops(labeled)
        print("Frontier num regions:", num)

        regions = [r for r in props if r.area >= min_area]
        regions = sorted(regions, key=lambda r: r.area, reverse=True)

        centers = []
        for i, region in enumerate(regions):
            if i >= max_centers:
                break
            cy, cx = region.centroid
            centers.append((int(round(cy)), int(round(cx))))

        print("Extracted centers:", centers)
        return centers




    def add_frontier_nodes_to_stg(self,G_stg, frontier_centers):
        """
        加入 frontier node 並與最近 STG 節點連結
        """
        nodes = list(G_stg.nodes)
        if not nodes:
            print("[add_frontier_nodes_to_stg] ⚠️ 原始G_stg沒有節點，跳過新增frontier。")
            return G_stg
        print("[add_frontier_nodes_to_stg] 即將加入frontier節點數：", len(frontier_centers))
        tree = cKDTree(np.array(nodes))
        for fpt in frontier_centers:
            dist, idx = tree.query(fpt)
            nearest_node = list(G_stg.nodes)[idx]
            G_stg.add_node(fpt, type='frontier')
            G_stg.add_edge(fpt, nearest_node, pixels=[fpt, nearest_node])

        return G_stg
    
    def mark_agent_node(self,G_stg, agent_pos):
        """
        將離 agent 最近的 junction node 改為 type='agent'
        """
        # 取出所有 junction 節點
        G_new = G_stg.copy()

        # 將舊的 agent node 還原為 junction
        for node, data in list(G_new.nodes(data=True)):
            if data.get("type") == "agent":
                G_new.nodes[node]["type"] = "junction"

        # 找出所有 junction 節點
        junction_nodes = [n for n, d in G_new.nodes(data=True) if d.get('type') == 'junction']
        if not junction_nodes:
            print("⚠️ Graph中沒有任何 junction，回傳空Graph。")
            return nx.Graph(), None


        # 找最近 junction
        tree = cKDTree(junction_nodes)
        dist, idx = tree.query(agent_pos)
        nearest_junction = junction_nodes[idx]

        # 複製其連線
        neighbors = list(G_new.neighbors(nearest_junction))
        edges_data = [G_new.get_edge_data(nearest_junction, n) for n in neighbors]

        # 移除原 junction
        G_new.remove_node(nearest_junction)

        # 新增 agent node
        G_new.add_node(agent_pos, type='agent')
        for neighbor, edata in zip(neighbors, edges_data):
            if edata and 'pixels' in edata:
                new_path = [agent_pos, neighbor]
                G_new.add_edge(agent_pos, neighbor, pixels=new_path)
            else:
                G_new.add_edge(agent_pos, neighbor)

        return G_new, nearest_junction
    
    def connect_components(self, G):
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if len(components) <=1:
            return G
        G_merged = nx.Graph()
        # 加入所有節點與邊
        for sub in components:
            G_merged.add_nodes_from(sub.nodes(data=True))
            G_merged.add_edges_from(sub.edges(data=True))
        # 用component中心連接
        centers = []
        for sub in components:
            coords = np.array(list(sub.nodes))
            center = tuple(np.mean(coords, axis=0).astype(int))
            centers.append(center)
        # 建立橋接邊
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                n1 = centers[i]
                n2 = centers[j]
                G_merged.add_edge(n1, n2, pixels=[n1, n2])
        return G_merged
    
    def remove_nodes_near_obstacles(self, G_stg, obs, min_distance=5):
        """
        移除距離障礙物太近的STG節點
        """
        obs_uint8 = (obs > 0.5).astype(np.uint8)
        dist_map = cv2.distanceTransform(1 - obs_uint8, cv2.DIST_L2, 3)
        nodes_to_remove = []
        for node in G_stg.nodes:
            y,x = node
            if dist_map[int(y), int(x)] < min_distance:
                nodes_to_remove.append(node)
        G_cleaned = G_stg.copy()
        G_cleaned.remove_nodes_from(nodes_to_remove)
        return G_cleaned



    def visualize_stg(self, occupancy_map,skeleton, G, frontiers):
        h, w = occupancy_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        color_map[occupancy_map == -1] = [150, 150, 150]
        color_map[occupancy_map == 0] = [255, 255, 255]
        color_map[occupancy_map == 100] = [255, 0, 0]
        color_map[frontiers == 1] = [255, 165, 0]
        # color_map[skeleton == 1] = [255, 165, 0] 
        plt.figure(figsize=(6, 6))
        plt.imshow(color_map)
        for (i, j), data in G.nodes(data=True):
            if data['type'] == 'junction':
                plt.plot(j, i, 'go', markersize=4)
            elif data['type'] == 'termination':
                plt.plot(j, i, 'ro', markersize=3)
            # elif data['type'] == 'frontier':
            #     plt.plot(j, i, 'm*', markersize=6)
            # elif data['type'] == 'agent':
            #     plt.plot(j, i, 'yX', markersize=8)
        for n1, n2 in G.edges():
            y1, x1 = n1
            y2, x2 = n2
            plt.plot([x1, x2], [y1, y2], 'orange', linewidth=1)
        if frontiers is not None:
            fy, fx = np.where(frontiers)
            plt.scatter(fx, fy, c='cyan', s=3, label='Frontiers')
        plt.title("Skeleton Topological Graph (STG)")
        plt.axis('off')
        plt.legend(loc='lower right')
        plt.show()

    def visualize_mst(self, occupancy_map, skeleton, G_mst):
        """
        視覺化MST（Minimum Spanning Tree）。
        """
        h, w = occupancy_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        color_map[occupancy_map == -1] = [150, 150, 150]
        color_map[occupancy_map == 0] = [255, 255, 255]
        color_map[occupancy_map == 100] = [255, 0, 0]
        # color_map[skeleton == 1] = [255, 165, 0]
        
        plt.figure(figsize=(6,6))
        
        # 畫節點
        for (i, j) in G_mst.nodes:
            plt.plot(j, i, 'bo', markersize=2)
        
        # 畫邊
        for (n1, n2) in G_mst.edges():
            y1, x1 = n1
            y2, x2 = n2
            plt.plot([x1, x2], [y1, y2], 'cyan', linewidth=1)

        plt.imshow(color_map, origin='lower')
        plt.title("Minimum Spanning Tree (MST)")
        plt.axis('off')
        plt.show()


# # Example usage with files
# if __name__ == "__main__":
#     builder = STGBuilder()
#     obs_files = sorted(glob.glob("maps_o/obstacle_20*_*.npy"))
#     exp_files = sorted(glob.glob("maps_e/explored_20*_*.npy"))
#     for i in range(len(obs_files)):
#         obs = np.load(obs_files[i])
#         exp = np.load(exp_files[i])
#         occ = builder.build_occupancy_map(obs, exp)
#         skel = builder.build_skeleton(obs, exp)
#         G_px = builder.skeleton_to_graph(skel)
#         G_px = nx.minimum_spanning_tree(G_px)
#         G_stg = builder.extract_stg(G_px)
#         G_stg = builder.merge_close_junctions(G_stg)
#         G_stg = builder.filter_stg_by_short_edges(G_stg)
#         builder.visualize_stg(occ, G_stg)
