import math
import heapq

# =============================
# 原始 GOODWIN 模型參數 (單位：公尺)
# =============================
SECTER_RADIUS_LEFT = 0.7 * 0.95
SECTER_RADIUS_RIGHT = 0.85 * 0.95
SECTER_RADIUS_BACK = 0.45 * 0.95


class n_fcc_a:
    """
    n_fcc_a 路徑規劃類別 (多干擾船版本)
    —— 原 fcc_a 改寫：不再考慮船速，均以一步一格（網格單位）計算，
       並且將干擾船擴展為多艘，皆以各自預先給定的 path 與 heading 做預測。

    參數：
      ship_info: dict，必須包含 'pos' 與 'goal' (單位：公尺)
      interfering_paths: list，每個元素為干擾船資訊 (dict)，格式：
         {
           "path": [(x,y), (x,y), ...],  # 以公尺表示，每步一格(與規劃船同步)
           "headings": [h0, h1, ...]     # 與 path 對應的航向(度)，務必給
         }
      grid_scale: 每格代表幾公尺 (例如 grid_scale=0.5 -> 一格=0.5m)

    在 calculate_path() 回傳：
      - 規劃船的路徑(公尺座標列表)
      - 規劃船每步的航向(headings)列表
    """

    def __init__(self, ship_info, interfering_paths, grid_scale=0.1, obstacles=None, time_obstacles=None):
        self.grid_scale = grid_scale

        # 簡易函式：從 pos 到 goal 計算初始航向
        def compute_heading(pos, goal):
            dx = goal[0] - pos[0]
            dy = goal[1] - pos[1]
            return math.degrees(math.atan2(dx, dy)) % 360

        # 1) 設定規劃 (yield) 船的起訖點（以「格」為單位）
        self.yield_pos = (
            int(round(ship_info["pos"][0] / grid_scale)),
            int(round(ship_info["pos"][1] / grid_scale)),
        )
        self.yield_goal = (
            int(round(ship_info["goal"][0] / grid_scale)),
            int(round(ship_info["goal"][1] / grid_scale)),
        )
        # 初始航向 (度)
        self.yield_heading = compute_heading(ship_info["pos"], ship_info["goal"])

        # 2) 讀入所有干擾船的 path 與 heading，並轉成「格」座標
        #    interfering_paths 其實是 array of dict
        self.interfering_paths = []
        for ip in interfering_paths:
            grid_path = []
            for px, py in ip["path"]:
                gx = int(round(px / grid_scale))
                gy = int(round(py / grid_scale))
                grid_path.append((gx, gy))

            # 以防 headings 與 path 長度不一致可加檢查 (此處略)
            # headings 直接保留原角度即可。
            self.interfering_paths.append(
                {
                    "grid_path": grid_path,
                    "headings": ip["headings"],
                }
            )
            
        # 3) 處理靜態障礙物資料，轉換成格座標並建立障礙物查詢表
        self.obstacles = []
        self.obstacle_lookup = {}  # 用於快速查詢某個座標是否在障礙物內
        
        if obstacles:
            for ox, oy, radius in obstacles:
                # 將障礙物座標和半徑轉換為格座標
                grid_ox = int(round(ox / grid_scale))
                grid_oy = int(round(oy / grid_scale))
                grid_radius = int(math.ceil(radius / grid_scale))
                
                self.obstacles.append((grid_ox, grid_oy, grid_radius))
                
                # 預先計算哪些格子是障礙物內部 (用查表方式加速路徑規劃)
                # 以矩形範圍包含整個圓形障礙物
                for x in range(grid_ox - grid_radius, grid_ox + grid_radius + 1):
                    for y in range(grid_oy - grid_radius, grid_oy + grid_radius + 1):
                        # 計算到障礙物中心的距離
                        dist = math.hypot(x - grid_ox, y - grid_oy)
                        # 如果距離小於等於半徑，則標記為障礙物
                        if dist <= grid_radius:
                            self.obstacle_lookup[(x, y)] = True
        
        # 4) 處理時間相關障礙物 (某時間步后才出現的障礙物)
        # time_obstacles 格式: [(x, y, radius, start_time), ...]
        self.time_obstacles = []
        self.time_obstacle_lookup = {}  # 格式: {(x, y, t): True} - 表示在時間t該點是障礙物
        
        if time_obstacles:
            for ox, oy, radius, start_time in time_obstacles:
                # 將障礙物座標和半徑轉換為格座標
                grid_ox = int(round(ox / grid_scale))
                grid_oy = int(round(oy / grid_scale))
                grid_radius = int(math.ceil(radius / grid_scale))
                
                self.time_obstacles.append((grid_ox, grid_oy, grid_radius, start_time))
                
                # 預先計算從start_time開始，哪些格子是障礙物內部
                for x in range(grid_ox - grid_radius, grid_ox + grid_radius + 1):
                    for y in range(grid_oy - grid_radius, grid_oy + grid_radius + 1):
                        # 計算到障礙物中心的距離
                        dist = math.hypot(x - grid_ox, y - grid_oy)
                        # 如果距離小於等於半徑，則標記為障礙物
                        if dist <= grid_radius:
                            # 注意：這裡我們是為每個時間步都建立查詢表
                            for t in range(start_time, 1000):  # 假設最多1000步
                                self.time_obstacle_lookup[(x, y, t)] = True

        # GOODWIN 模型參數換算為「格」單位
        self.g_secter_radius_left = SECTER_RADIUS_LEFT / grid_scale
        self.g_secter_radius_right = SECTER_RADIUS_RIGHT / grid_scale
        self.g_secter_radius_back = SECTER_RADIUS_BACK / grid_scale

        self.TWICE_SECTER_RADIUS_MAX = 2 * self.g_secter_radius_right
        self.TWICE_SECTER_RADIUS_MIN = 2 * self.g_secter_radius_back
        self.INTERVAL_PARAMETER_67_180 = (
            self.g_secter_radius_left + self.g_secter_radius_right
        )
        self.INTERVAL_PARAMETER_245_360 = (
            self.g_secter_radius_left + self.g_secter_radius_back
        )
        self.INTERVAL_PARAMETER_0_67 = (
            self.g_secter_radius_right + self.g_secter_radius_back
        )

        # 同樣根據 grid_scale 設定 FCC_Scale
        self.fcc_scale =  grid_scale*20
        self.heading_history_length = 4  # 可調參數

        # 3) 定義 A* 搜索區域
        #    包含：yield 船起點、終點，再把所有干擾船走過的點都考慮進去，最後加上一些 margin
        xs = [self.yield_pos[0], self.yield_goal[0]]
        ys = [self.yield_pos[1], self.yield_goal[1]]
        for ip in self.interfering_paths:
            for xx, yy in ip["grid_path"]:
                xs.append(xx)
                ys.append(yy)

        margin = 20
        self.min_x = min(xs) - margin
        self.max_x = max(xs) + margin
        self.min_y = min(ys) - margin
        self.max_y = max(ys) + margin

        # 內部分析用
        self.analysis = {}

    # ----------------------------------------------------------------
    # 保留原先的兩船 Goodwin FCC 模型函式，計算 u_theta, u_dist, fcc
    # ----------------------------------------------------------------
    def calc_u_theta(self, theta1, theta2):
        angle_diff = abs(theta1 - theta2)
        return (17 / 44) * (
            math.cos(math.radians(angle_diff - 19))
            + math.sqrt(440 / 289 + math.cos(math.radians(angle_diff - 19)) ** 2)
        )

    def calc_u_dist(self, dist, theta1, theta2):
        if dist > self.TWICE_SECTER_RADIUS_MAX:
            return 0
        elif dist < self.TWICE_SECTER_RADIUS_MIN:
            return 1
        delta = abs(theta1 - theta2)
        if 67.5 <= delta < 180:
            return (
                (self.INTERVAL_PARAMETER_67_180 - dist) / self.INTERVAL_PARAMETER_67_180
            ) ** 2
        elif 247.5 <= delta < 360:
            return (
                (self.INTERVAL_PARAMETER_245_360 - dist)
                / self.INTERVAL_PARAMETER_245_360
            ) ** 2
        else:
            return (
                (self.INTERVAL_PARAMETER_0_67 - dist) / self.INTERVAL_PARAMETER_0_67
            ) ** 2

    def calc_fcc(self, dist, theta1, theta2):
        return 0.5 * self.calc_u_theta(theta1, theta2) + 0.5 * self.calc_u_dist(
            dist, theta1, theta2
        )

    # ---------------------------
    # 依需求保留，供主程式測試用 (移動一格為一步)
    # ---------------------------
    def predict_direct_path(self):
        """
        參考函式，用於示範如何一步一格地從 pos 走到 goal
        （若您要測試單艘干擾船，可用此產生它的 path）
        """
        # 這裡只是保留供測試，不會在本類別內直接用
        pos = [self.yield_pos[0], self.yield_pos[1]]  # 這裡暫時當範例
        goal = self.yield_goal
        path = [tuple(pos)]
        while (pos[0], pos[1]) != goal:
            dx = goal[0] - pos[0]
            dy = goal[1] - pos[1]
            step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
            step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
            pos[0] += step_x
            pos[1] += step_y
            path.append(tuple(pos))
        return path

    # ---------------------------
    # 八方向距離估計
    # ---------------------------
    @staticmethod
    def h_cost_distance(a, b):
        (x1, y1) = a
        (x2, y2) = b
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return (math.sqrt(2) - 1) * min(dx, dy) + max(dx, dy)

    # ---------------------------
    # A* 中要用的啟發式 (_heuristic)
    # 這裡把所有干擾船「同時間 t」的 FCC 都加總
    # ---------------------------
    def _heuristic(self, state):
        x, y, t, curr_h, _ = state
        # 1) 對目標的距離
        dx_goal = abs(x - self.yield_goal[0])
        dy_goal = abs(y - self.yield_goal[1])
        h_dist = (math.sqrt(2) - 1) * min(dx_goal, dy_goal) + max(dx_goal, dy_goal)

        # 2) 估計 FCC(針對所有干擾船)
        sum_fcc_cost = 0
        for ip in self.interfering_paths:
            grid_path = ip["grid_path"]
            headings = ip["headings"]
            # 若 t 超過此干擾船的 path，則視為停在最後一格
            if t < len(grid_path):
                other_pos = grid_path[t]
                other_head = headings[t]
            else:
                other_pos = grid_path[-1]
                other_head = headings[-1]

            # 計算與規劃船位置的距離、方位
            dx = x - other_pos[0]
            dy = y - other_pos[1]
            D = math.hypot(dx, dy)
            if D == 0:
                # 若重疊，就給個預設角度
                yield_theta = 0
            else:
                # 定義：正北為 0，順時針增加
                yield_theta = math.degrees(math.atan2(dx, dy)) % 360

            # 單艘干擾船的 fcc
            fcc_val = self.calc_fcc(D, yield_theta, other_head)
            sum_fcc_cost += fcc_val

        # 總和 FCC 後再乘以縮放
        return h_dist + self.fcc_scale * sum_fcc_cost

    # ---------------------------
    # 產生鄰居：允許 8 方向移動，但避開障礙物
    # ---------------------------
    def _neighbors(self, state):
        x, y, t, curr_h, h_history = state  # h_history: tuple of previous headings
        nbrs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                # 計算新位置
                new_x = x + dx
                new_y = y + dy
                new_t = t + 1
                
                # 檢查是否在靜態障礙物內部 (利用快速查表)
                if (new_x, new_y) in self.obstacle_lookup:
                    continue
                
                # 檢查是否在時間相關障礙物內部 (根據當前時間t)
                if (new_x, new_y, new_t) in self.time_obstacle_lookup:
                    continue
                
                # 計算候選航向
                cand_h = math.degrees(math.atan2(dx, dy)) % 360

                # 與歷史航向比對，若有任一差異 >= 90° 則捨棄此候選
                valid = True
                for past_h in h_history:
                    diff = abs((cand_h - past_h + 180) % 360 - 180)
                    if diff >= 90:
                        valid = False
                        break
                if not valid:
                    continue

                # 更新航向歷史：若未滿預設長度則直接加；否則捨棄最舊的
                if len(h_history) < self.heading_history_length:
                    new_history = h_history + (cand_h,)
                else:
                    new_history = h_history[1:] + (cand_h,)

                nbrs.append((new_x, new_y, new_t, cand_h, new_history))
        return nbrs

    # ---------------------------
    # A* 搜索主體
    # ---------------------------
    def _a_star(self):
        start_state = (
            self.yield_pos[0],
            self.yield_pos[1],
            0,
            self.yield_heading,
            (self.yield_heading,),  # 初始歷史只有起始航向
        )
        goal_xy = (self.yield_goal[0], self.yield_goal[1])

        open_heap = []
        g_cost = {start_state: 0}
        parent = {}
        f_start = self._heuristic(start_state)
        heapq.heappush(open_heap, (f_start, start_state))
        closed = set()

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            if (current[0], current[1]) == goal_xy:
                # 回溯路徑
                path = []
                state = current
                while state in parent:
                    path.append(state)
                    state = parent[state]
                path.append(start_state)
                path.reverse()
                return path

            closed.add(current)
            for neighbor in self._neighbors(current):
                if neighbor in closed:
                    continue
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                step_cost = math.sqrt(2) if (dx != 0 and dy != 0) else 1
                tentative_g = g_cost[current] + step_cost

                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    parent[neighbor] = current
                    f_val = tentative_g + self._heuristic(neighbor)
                    heapq.heappush(open_heap, (f_val, neighbor))

        return []

    # ---------------------------
    # 主函式：計算規劃船的路徑
    # 回傳 (路徑座標(公尺), 每步航向列表)
    # ---------------------------
    def calculate_path(self):
        yield_path_states = self._a_star()  # (x, y, t, heading)
        if not yield_path_states:
            return [], []

        # 只取格子座標 (x,y)
        yield_path_grid = [(s[0], s[1]) for s in yield_path_states]

        # 轉成公尺
        yield_path_m = [
            (x * self.grid_scale, y * self.grid_scale) for (x, y) in yield_path_grid
        ]

        # 分析過程 (時間步、位置、heading)
        analysis_positions = yield_path_m
        analysis_steps = [s[2] for s in yield_path_states]

        # 估算規劃船在每個步驟的航向(度)
        # (此處做個簡單的近似：用前 k_step 筆來估計當前 heading)
        k_step = 3
        computed_headings = []
        for i in range(len(analysis_positions)):
            if i == 0:
                # 第一點就給初始航向
                computed_headings.append(self.yield_heading)
            else:
                j = max(0, i - k_step)
                start_pos = analysis_positions[j]
                pos = analysis_positions[i]
                dx = pos[0] - start_pos[0]
                dy = pos[1] - start_pos[1]
                dist = math.hypot(dx, dy)
                if dist < 1e-6:
                    # 幾乎沒移動，就沿用上一個
                    computed_headings.append(computed_headings[-1])
                else:
                    head_angle = math.degrees(math.atan2(dx, dy)) % 360
                    computed_headings.append(head_angle)

        # 計算 FCC (把所有干擾船在同一步的干擾都加總)
        computed_fcc = []
        for i, pos in enumerate(analysis_positions):
            sum_fcc_i = 0
            yield_head_i = computed_headings[i]
            for ip in self.interfering_paths:
                grid_path = ip["grid_path"]
                headings = ip["headings"]
                if i < len(grid_path):
                    other_pos_grid = grid_path[i]
                    other_head = headings[i]
                else:
                    other_pos_grid = grid_path[-1]
                    other_head = headings[-1]

                # 距離需用公尺
                dx = pos[0] - (other_pos_grid[0] * self.grid_scale)
                dy = pos[1] - (other_pos_grid[1] * self.grid_scale)
                D = math.hypot(dx, dy)

                # 計算 FCC
                sum_fcc_i += self.calc_fcc(D, yield_head_i, other_head)

            computed_fcc.append(sum_fcc_i * self.fcc_scale)

        # 存在 self.analysis 以供外部調用或畫圖
        self.analysis = {
            "steps": analysis_steps,
            "positions": analysis_positions,
            "headings": computed_headings,
            "fcc": computed_fcc,
        }

        # 回傳路徑(公尺)及對應的航向列表
        return yield_path_m, computed_headings


# =============================================================================
# 測試及視覺化（使用 pygame 畫出路徑圖與嵌入 Matplotlib 分析圖）
# =============================================================================
if __name__ == "__main__":
    import pygame
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import os

    # 設定 Pygame 視窗位置
    os.environ["SDL_VIDEO_WINDOW_POS"] = "100,200"

    # ---------------------------
    # 輔助函式：畫箭頭（以三角形表示）
    # ---------------------------
    def draw_arrow(surface, color, center, angle, size):
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
        # 這裡示範「一艘要規劃的船」+「一艘干擾船」+「障礙物」
        # ---------------------------
        # Yield 船的起點、終點 (公尺)
        yield_ship_info = {
            "pos": (0, 50),
            "goal": (50, 0),
        }

        # 干擾船(原本的直行船)：
        # 這裡用「predict_direct_path」邏輯來產生一條路徑 + 對應 heading 做範例
        # --------------------------------------------------------------------
        # 假設干擾船起點 (0,0), 目標 (10,10)，一步一格(對應 grid_scale=0.2 共走 50步)
        # 注意: 干擾船的 path/headings 單位是「公尺」+「角度」
        # --------------------------------------------------------------------
        def gen_interfering_example(start, goal, grid_scale=0.2):
            # 先轉「格座標」來用 predict 的邏輯
            sx = int(round(start[0] / grid_scale))
            sy = int(round(start[1] / grid_scale))
            gx = int(round(goal[0] / grid_scale))
            gy = int(round(goal[1] / grid_scale))
            path_grid = []
            pos = [sx, sy]
            path_grid.append((sx, sy))
            while (pos[0], pos[1]) != (gx, gy):
                dx = gx - pos[0]
                dy = gy - pos[1]
                step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
                step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
                pos[0] += step_x
                pos[1] += step_y
                path_grid.append((pos[0], pos[1]))

            # 還原成公尺
            path_m = [(p[0] * grid_scale, p[1] * grid_scale) for p in path_grid]

            # 每步 heading
            headings = []
            prev = None
            for i, p in enumerate(path_m):
                if i == 0:
                    dx_0 = path_m[-1][0] - p[0]
                    dy_0 = path_m[-1][1] - p[1]
                    # 順便來個預設(這裡也可用任何邏輯)
                    head_0 = math.degrees(math.atan2(dx_0, dy_0)) % 360
                    headings.append(head_0)
                    prev = p
                else:
                    dx_1 = p[0] - prev[0]
                    dy_1 = p[1] - prev[1]
                    dist_1 = math.hypot(dx_1, dy_1)
                    if dist_1 < 1e-6:
                        headings.append(headings[-1])
                    else:
                        head_1 = math.degrees(math.atan2(dx_1, dy_1)) % 360
                        headings.append(head_1)
                    prev = p
            return path_m, headings

        # 產生干擾船路徑資料
        interfering_path_m, interfering_headings = gen_interfering_example(
            start=(0, 0), goal=(50, 50), grid_scale=1
        )
        interfering_paths = [
            {
                "path": interfering_path_m,
                "headings": interfering_headings,
            }
        ]

        # 新增一艘干擾船 (0, 5) -> (10, 5)
        interfering_path_m2, interfering_headings2 = gen_interfering_example(
            start=(0, 0), goal=(50, 50), grid_scale=1        )
        interfering_paths.append(
            {
                "path": interfering_path_m2,
                "headings": interfering_headings2,
            }
        )
        
        # 定義障礙物 (中心x, 中心y, 半徑) - 單位公尺
        obstacles = [
             # 中心點 (5,5), 半徑 1.5m
            (3, 7, 0.8),     # 中心點 (3,7), 半徑 0.8m
        ]

        # 建立 n_fcc_a 物件
        planner = n_fcc_a(
            ship_info=yield_ship_info,
            interfering_paths=interfering_paths,
            grid_scale=1,
            obstacles=obstacles,  # 添加障礙物
        )

        # 計算路徑
        yield_path_m, yield_headings = planner.calculate_path()
        analysis = planner.analysis
        yield_steps = analysis["steps"]
        yield_positions = analysis["positions"]  # (公尺)
        fcc_values = analysis["fcc"]

        # ---------------------------
        # Pygame 視覺化設定
        # ---------------------------
        pygame.init()
        desired_draw_area_width = 1200
        desired_draw_area_height = 1200

        # 合併兩者路徑的 x,y 以計算可視化範圍
        all_x = [pt[0] for pt in yield_positions] + [pt[0] for pt in interfering_path_m]
        all_y = [pt[1] for pt in yield_positions] + [pt[1] for pt in interfering_path_m]
        margin_m = 3
        min_x = int(min(all_x)) - margin_m
        max_x = int(max(all_x)) + margin_m
        min_y = int(min(all_y)) - margin_m
        max_y = int(max(all_y)) + margin_m

        m_range_x = max_x - min_x
        m_range_y = max_y - min_y
        vis_scale = min(
            desired_draw_area_width / m_range_x, desired_draw_area_height / m_range_y
        )
        ship_path_width = m_range_x * vis_scale
        ship_path_height = m_range_y * vis_scale

        plot_width = 600
        total_width = int(ship_path_width + plot_width)
        total_height = int(ship_path_height)

        screen = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("n_fcc_a 多船FCC路徑測試")
        font = pygame.font.SysFont(None, 16)
        clock = pygame.time.Clock()

        # 產生 Matplotlib 圖：FCC vs Steps (亦可觀察 Heading vs Steps)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].plot(yield_steps, fcc_values, "c.-")
        axs[0].set_title("FCC vs Steps (Yield 船)")
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("FCC")
        axs[0].grid(True)

        axs[1].plot(yield_steps, yield_headings, "b.-")
        axs[1].set_title("Heading vs Steps")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Heading (度)")
        axs[1].grid(True)
        plt.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = fig.canvas.get_width_height()
        plot_surface = pygame.image.fromstring(raw_data, size, "RGB")
        fig_w, fig_h = size
        aspect_ratio = fig_w / fig_h
        desired_w = plot_width
        desired_h = int(desired_w / aspect_ratio)
        plot_surface = pygame.transform.scale(plot_surface, (desired_w, desired_h))

        def trans_to_screen(px, py):
            # 將 (公尺) 轉為畫面座標
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

            # 畫網格
            for gx in range(min_x, max_x + 1):
                xx = (gx - min_x) * vis_scale
                pygame.draw.line(screen, (220, 220, 220), (xx, 0), (xx, ship_path_height))
            for gy in range(min_y, max_y + 1):
                yy = ship_path_height - ((gy - min_y) * vis_scale)
                pygame.draw.line(screen, (220, 220, 220), (0, yy), (ship_path_width, yy))
                
            # 繪製障礙物 (以深灰色圓形表示)
            for ox, oy, radius in obstacles:
                sx, sy = trans_to_screen(ox, oy)
                screen_radius = int(radius * vis_scale)
                pygame.draw.circle(screen, (80, 80, 80), (int(sx), int(sy)), screen_radius)
                # 繪製障礙物中心點和標籤
                pygame.draw.circle(screen, (0, 0, 0), (int(sx), int(sy)), 3)
                label = font.render(f"障礙物 ({ox},{oy})", True, (0, 0, 0))
                screen.blit(label, (int(sx) + 5, int(sy) - 15))

            # 畫 Yield 船路徑 (藍色)，並於每步畫箭頭
            for i, (xm, ym) in enumerate(yield_positions):
                sx, sy = trans_to_screen(xm, ym)
                draw_arrow(screen, (0, 0, 255), (sx, sy), yield_headings[i], 10)
                txt = font.render(str(yield_steps[i]), True, (0, 0, 255))
                screen.blit(txt, (sx + 2, sy + 2))

            # 畫所有干擾船路徑 (紅色)
            for ip_data in interfering_paths:
                direct_steps = list(range(len(ip_data["path"])))
                direct_headings = ip_data["headings"]
                for i, (xm, ym) in enumerate(ip_data["path"]):
                    sx, sy = trans_to_screen(xm, ym)
                    draw_arrow(screen, (255, 0, 0), (sx, sy), direct_headings[i], 10)
                    txt = font.render(str(direct_steps[i]), True, (255, 0, 0))
                    screen.blit(txt, (sx + 2, sy + 2))

            # 貼上 Matplotlib 圖
            screen.blit(plot_surface, (int(ship_path_width), 0))
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    main()
