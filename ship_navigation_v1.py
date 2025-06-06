import math
import numpy as np
from scipy.interpolate import splprep, splev
from multi_ship_planner_v1 import n_fcc_a


def distance(p1, p2):
    """計算 p1, p2 之間的歐幾里得距離"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_position_at_step(path, step_index):
    """
    取得 path 中某個 step 的位置:
      - step_index < len(path) -> 直接回傳 path[step_index]
      - step_index >= len(path) -> 回傳 path[-1] (最後一個位置)
    """
    if step_index < len(path):
        return path[step_index]
    else:
        return path[-1]


def paths_conflict(pathA, pathB, safe_distance):
    """
    檢查 pathA 與 pathB 是否在「同時間步(step)」發生衝突:
      - 距離 < safe_distance 則判定有衝突
      - 若最長路徑長度為 max_len，則比對 step=0 ~ max_len-1。
        超出範圍的部分，用該船最後一個位置。
    """
    max_steps = max(len(pathA), len(pathB))
    for step in range(max_steps):
        posA = get_position_at_step(pathA, step)
        posB = get_position_at_step(pathB, step)
        if distance(posA, posB) < safe_distance:
            return True
    return False


# --------------------
# 平滑化相關函式
# --------------------
def moving_average_smooth(path, window_size=4):
    """
    移動平均平滑化，輸出與輸入路徑相同長度的結果
    """
    if len(path) < window_size:
        return path[:]
    smoothed = []
    half_window = window_size // 2
    for i in range(len(path)):
        indices = range(max(0, i - half_window), min(len(path), i + half_window + 1))
        avg_x = np.mean([path[j][0] for j in indices])
        avg_y = np.mean([path[j][1] for j in indices])
        smoothed.append((avg_x, avg_y))
    return smoothed


def bezier_smooth(path, smooth_factor=0.1):
    """
    利用樣條平滑化，輸出與輸入路徑相同長度的結果
    smooth_factor 對應於 scipy.interpolate.splprep 的 s 參數
    """
    if len(path) < 3:
        return path[:]
    x, y = zip(*path)
    tck, u = splprep([x, y], s=smooth_factor)
    # 取與原始點數相同的參數點
    u_new = np.linspace(0, 1, len(path))
    x_new, y_new = splev(u_new, tck)
    return list(zip(x_new, y_new))


def recalc_headings(path):
    """
    根據路徑計算每個步驟的 heading (度)
    heading[i] = 從 path[i] 指向 path[i+1] 的角度，最後一點延用前一點
    """
    headings = []
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        heading_deg = math.degrees(math.atan2(dy, dx))
        headings.append(heading_deg)
    if len(path) > 1:
        headings.append(headings[-1])
    else:
        headings.append(0.0)
    return headings


def smooth_path(path, method="none"):
    """
    依據 method 平滑化路徑，保持輸出長度與輸入一致
      method:
        - "none"：不平滑化
        - "moving_average"：移動平均平滑化，param 為 window_size
        - "bezier"：樣條平滑化，param 為 smooth_factor (s)
    回傳 (平滑後路徑, 重新計算的 headings)
    """
    if len(path) <= 1:
        return path[:], [0.0] * len(path)

    if method == "none":
        new_path = path[:]
    elif method == "moving_average":
        new_path = moving_average_smooth(path)
    elif method == "bezier":
        new_path = bezier_smooth(path)
    else:
        new_path = path[:]

    new_headings = recalc_headings(new_path)
    return new_path, new_headings


# --------------------
# 多船規劃主函式
# --------------------
def multi_ship_planning(
    ships,
    safe_distance=5,
    grid_scale=1,
    smoothing_method="none",
    obstacles=None,  # 靜態障礙物參數，格式: [(x, y, radius), ...]
    time_obstacles=None,  # 時間相關障礙物參數，格式: [(x, y, radius, start_time), ...]
    goal_obstacle_radius=2.5,  # 新增: 目標障礙物半徑
):
    """
    ships: list，裡面每個元素是一艘船的資訊，結構例如：
      [
        {"id":"A", "pos": (x1,y1), "goal": (gx1,gy1)},
        {"id":"B", "pos": (x2,y2), "goal": (gx2,gy2)},
        ...
      ]
    obstacles: list，每個元素為 (x, y, radius)，表示障礙物中心座標和半徑(公尺)
    time_obstacles: list，每個元素為 (x, y, radius, start_time)，表示在start_time時間步後才出現的障礙物

    規則：
      1. 先計算所有船到目的地的直線距離，距離最長的先規劃
      2. 規劃每艘船時，先嘗試不加任何干擾路徑。若發現與前面船的路徑有衝突(距離 < safe_distance)，
         則將「有衝突的船路徑」加入 interfering_paths，再重算。反覆至無新衝突為止。
      3. 完成後將結果存入 planning_results；並將新船的路徑存入 previous_planned_paths 以供後續比對。
      4. 若提供障礙物資訊，路徑規劃將避開所有障礙物。
      5. 對於時間相關障礙物，僅在特定時間步後才會考慮避開。

    回傳:
      {
        "船id或index": {
          "path": [(x,y), (x,y), ...],   # 船之路徑(公尺)
          "headings": [h1, h2, ...],       # 對應之航向(度)
          "analysis": {...}                # (可選) n_fcc_a.analysis 的資訊
        },
        ...
      }
    """
    # 1. 依直線距離排序 (遠->近)
    sorted_ships = sorted(
        ships, key=lambda s: distance(s["pos"], s["goal"]), reverse=True
    )

    # 如果沒有提供障礙物，初始化為空列表
    if obstacles is None:
        obstacles = []
    
    # 如果沒有提供時間相關障礙物，初始化為空列表
    if time_obstacles is None:
        time_obstacles = []

    # 用來存最終結果
    planning_results = {}

    # 用來存已經「確定」的路徑，以供後續船做衝突比對
    previous_planned_paths = []
    
    # 用來存儲已規劃船隻的到達時間，以便創建時間相關障礙物
    ship_arrival_obstacles = []

    # --- 新增: 目標障礙物自動處理 ---
    # 先建立所有目標障礙物(排除會困住起點/其他目標的)
    goal_obstacles = []
    for ship in sorted_ships:
        gx, gy = ship["goal"]
        sid = ship["id"]
        blocked = False
        for other in sorted_ships:
            if distance((gx, gy), other["pos"]) < goal_obstacle_radius:
                blocked = True
                break
        if not blocked:
            for other in sorted_ships:
                if other["id"] != sid and distance((gx, gy), other["goal"]) < goal_obstacle_radius:
                    blocked = True
                    break
        if not blocked:
            goal_obstacles.append((gx, gy, goal_obstacle_radius, sid))

    # --- 逐艘規劃 ---
    for ship in sorted_ships:
        ship_id = ship.get("id", None)
        if not ship_id:
            ship_id = f"{ship['pos']}->{ship['goal']}"
        print(f"開始規劃船 {ship_id} ...")
        # 除了自己的目標，其餘都設為障礙物
        other_goal_obstacles = [(gx, gy, r) for (gx, gy, r, sid) in goal_obstacles if sid != ship_id]
        all_obstacles = obstacles + other_goal_obstacles
        ship_info = {"pos": ship["pos"], "goal": ship["goal"]}
        interfering_paths = []
        all_time_obstacles = time_obstacles + ship_arrival_obstacles
        while True:
            planner = n_fcc_a(
                ship_info=ship_info,
                interfering_paths=interfering_paths,
                grid_scale=grid_scale,
                obstacles=all_obstacles,
                time_obstacles=all_time_obstacles,
            )
            path_m, headings = planner.calculate_path()
            new_conflict_found = False
            for prev_ship in previous_planned_paths:
                if prev_ship not in interfering_paths:
                    if paths_conflict(path_m, prev_ship["path"], safe_distance):
                        print(f"  - 發現與船 {prev_ship['id']} 衝突，加入干擾重算。")
                        interfering_paths.append(prev_ship)
                        new_conflict_found = True
            if not new_conflict_found:
                break
        print(
            f"船 {ship_id} 規劃完成！路徑長度={len(path_m)} 步\n"
            f"參照路徑: {', '.join([str(p['id']) for p in interfering_paths])}\n"
        )
        
        # 儲存這艘船的路徑和分析結果
        planning_results[ship_id] = {
            "path": path_m,
            "headings": headings,
            "analysis": planner.analysis,
        }
        
        # 將這艘船加入已規劃的船隻列表，供後續船隻參考
        previous_planned_paths.append(
            {"id": ship_id, "path": path_m, "headings": headings}
        )
        
        # 若分析結果中有時間步資訊，將該船到達終點的位置及時間作為時間相關障礙物
        if path_m and "steps" in planner.analysis:
            arrival_time = planner.analysis["steps"][-1]  # 到達終點的時間步
            arrival_pos = path_m[-1]  # 終點位置(公尺)
            # 添加到時間相關障礙物列表，半徑設為5公尺
            ship_arrival_obstacles.append((arrival_pos[0], arrival_pos[1], 5.0, arrival_time))
            print(f"  - 船 {ship_id} 將在時間步 {arrival_time} 到達終點 {arrival_pos}，設為障礙區域")

    # 3. 規劃完所有船後，再對所有結果進行平滑化（smoothing_method=="none"則直接保持原狀）
    for ship_id in planning_results:
        raw_path, raw_headings = (
            planning_results[ship_id]["path"],
            planning_results[ship_id]["headings"],
        )
        if smoothing_method != "none":
            new_path, new_headings = smooth_path(raw_path, method=smoothing_method)
        else:
            new_path, new_headings = raw_path, raw_headings
        planning_results[ship_id]["path"] = new_path
        planning_results[ship_id]["headings"] = new_headings

    # 預設只回傳 planning_results，goal_obstacles 僅內部使用
    return planning_results
    # 若未來需 goal_obstacles，可考慮加參數控制是否一併回傳


# --------------------
# Pygame 視覺化
# --------------------
import sys
import pygame
import random


def draw_arrow(surface, color, center, angle, size=10):
    """
    使用三角形繪製箭頭，表示航向
    """
    rad = math.radians(angle)
    tip = (center[0] + size * math.sin(rad), center[1] - size * math.cos(rad))
    side_angle = math.radians(150)
    left = (
        center[0] + (size * 0.5) * math.sin(rad + side_angle),
        center[1] - (size * 0.5) * math.cos(rad + side_angle),
    )
    right = (
        center[0] + (size * 0.5) * math.sin(rad - side_angle),
        center[1] - (size * 0.5) * math.cos(rad - side_angle),
    )
    pygame.draw.polygon(surface, color, [tip, left, right])


def main():
    # ---------------------------
    # 定義多艘船 (示範資料)
    # ---------------------------
    ships_data = [
        {"id": "ShipA", "pos": (5.06, 3.45), "goal": (40.02, 22.31)},
        {"id": "ShipB", "pos": (11.73, 3.45), "goal": (35.19, 20.01)},
        {"id": "ShipC", "pos": (18.17, 3.45), "goal": (33.81, 28.52)},
        {"id": "ShipD", "pos": (18.17, -3.22), "goal": (31.51, 23.69)},
        {"id": "ShipE", "pos": (18.17, -9.66), "goal": (39.33, 27.83)},
    ]
    # ---------------------------
    # 目標障礙物 (檢查不困住起點/其他目標)
    # ---------------------------
    goal_obstacles = []
    goal_radius = 2.5
    for ship in ships_data:
        gx, gy = ship["goal"]
        sid = ship["id"]
        blocked = False
        # 檢查所有船的起點是否在障礙物範圍內
        for other in ships_data:
            if distance((gx, gy), other["pos"]) < goal_radius:
                blocked = True
                break
        # 檢查其他目標點是否在障礙物範圍內（不含自己）
        if not blocked:
            for other in ships_data:
                if other["id"] != sid and distance((gx, gy), other["goal"]) < goal_radius:
                    blocked = True
                    break
        if not blocked:
            goal_obstacles.append((gx, gy, goal_radius, sid))
    # ---------------------------
    # 呼叫多船規劃，只取得所有結果（不再解包 goal_obstacles）
    # ---------------------------
    results = multi_ship_planning(
        ships_data,
        safe_distance=3,
        grid_scale=0.5,
        smoothing_method="none",
    )

    # ---------------------------
    # Pygame 視覺化設定
    # ---------------------------
    pygame.init()
    desired_draw_area_width = 1200
    desired_draw_area_height = 800

    # 整合所有船路徑點，決定顯示範圍
    all_x = []
    all_y = []
    for sid, data in results.items():
        for px, py in data["path"]:
            all_x.append(px)
            all_y.append(py)
    margin_m = 3
    min_x = int(min(all_x)) - margin_m
    max_x = int(max(all_x)) + margin_m
    min_y = int(min(all_y)) - margin_m
    max_y = int(max(all_y)) + margin_m

    m_range_x = max_x - min_x
    m_range_y = max_y - min_y
    if m_range_x < 1e-6:
        m_range_x = 1
    if m_range_y < 1e-6:
        m_range_y = 1

    vis_scale = min(
        desired_draw_area_width / m_range_x, desired_draw_area_height / m_range_y
    )
    ship_path_width = m_range_x * vis_scale
    ship_path_height = m_range_y * vis_scale

    screen = pygame.display.set_mode((int(ship_path_width), int(ship_path_height)))
    pygame.display.set_caption("多船 FCC 路徑規劃")
    font = pygame.font.SysFont(None, 16)
    clock = pygame.time.Clock()

    # 為每艘船指定隨機顏色
    color_map = {}
    for sid in results.keys():
        color_map[sid] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )

    def trans_to_screen(px, py):
        # 將公尺座標轉換為螢幕座標
        sx = (px - min_x) * vis_scale
        sy = ship_path_height - ((py - min_y) * vis_scale)
        return sx, sy

    running = True
    while running:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        # 先畫所有目標障礙物 (半透明藍色，最下層)
        for ox, oy, radius, sid in goal_obstacles:
            sx, sy = trans_to_screen(ox, oy)
            # 建立一個帶 alpha 的 surface 畫圓
            circle_surface = pygame.Surface((int(radius*2*vis_scale), int(radius*2*vis_scale)), pygame.SRCALPHA)
            pygame.draw.circle(circle_surface, (0, 128, 255, 80), (int(radius*vis_scale), int(radius*vis_scale)), int(radius * vis_scale))
            screen.blit(circle_surface, (int(sx-radius*vis_scale), int(sy-radius*vis_scale)))
            # 外框
            pygame.draw.circle(screen, (0, 64, 180), (int(sx), int(sy)), int(radius * vis_scale), 2)
            txt = font.render(f"{sid}", True, (0, 64, 180))
            screen.blit(txt, (sx + 2, sy - 10))

        # 畫網格
        for gx in range(min_x, max_x + 1):
            xx = (gx - min_x) * vis_scale
            pygame.draw.line(screen, (220, 220, 220), (xx, 0), (xx, ship_path_height))
        for gy in range(min_y, max_y + 1):
            yy = ship_path_height - ((gy - min_y) * vis_scale)
            pygame.draw.line(screen, (220, 220, 220), (0, yy), (ship_path_width, yy))

        # 畫每艘船的航跡與步數
        for sid, data in results.items():
            path = data["path"]
            headings = data["headings"]
            col = color_map[sid]
            for i, (px, py) in enumerate(path):
                sx, sy = trans_to_screen(px, py)
                draw_arrow(screen, col, (sx, sy), headings[i], 10)
                step_txt = font.render(str(i), True, col)
                screen.blit(step_txt, (sx + 2, sy + 2))

        # 在右上角繪製 legend (顯示船 id 與顏色)
        legend_x = 10
        legend_y = int(ship_path_width) - 150
        legend_width = 140
        legend_height = 20 * len(results) + 10
        pygame.draw.rect(
            screen, (240, 240, 240), (legend_x, legend_y, legend_width, legend_height)
        )
        pygame.draw.rect(
            screen, (0, 0, 0), (legend_x, legend_y, legend_width, legend_height), 1
        )
        for i, sid in enumerate(results.keys()):
            text = font.render(sid, True, color_map[sid])
            screen.blit(text, (legend_x + 5, legend_y + 5 + i * 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
