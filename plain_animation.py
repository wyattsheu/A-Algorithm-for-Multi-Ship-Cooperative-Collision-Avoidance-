import sys
import math
import pygame
import random
from pygame.locals import K_LSHIFT, K_RSHIFT

# -----------------------------------------------------------------------------
# 多船規劃模組匯入
# 假設 ship_navigation_v1.py 與本程式在同目錄，可直接 import
# -----------------------------------------------------------------------------
from ship_navigation_v1 import multi_ship_planning


def draw_arrow(surface, color, center, angle, size=10):
    """
    使用三角形繪製箭頭表示航向。

    參數:
      surface (pygame.Surface): 畫布對象。
      color (tuple): 顏色 (R, G, B)。
      center (tuple): 箭頭中心座標 (x, y)。
      angle (float): 航向角度（以度為單位）。
      size (int): 箭頭大小。
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


class AnimatedShip:
    """
    動畫模式用的船物件。

    屬性:
      ship_id (str): 船的 ID。
      path (list): 路徑，格式為 [(x, y), ...]（單位：公尺）。
      speed (float): 每幀移動的距離（公尺）。
      pos (tuple): 當前位置（公尺）。
      heading (float): 當前航向（度），0 表示正北，順時針增加。
      target_waypoint_index (int): 目標 waypoint 的索引。
      done (bool): 是否已經走完路徑。
    """

    def __init__(self, ship_id, path, speed=0.05):
        """
        初始化 AnimatedShip 物件。

        參數:
          ship_id (str): 船的 ID。
          path (list): 航跡路徑（公尺座標列表）。
          speed (float): 每幀移動的距離，預設 0.05 公尺。
        """
        self.ship_id = ship_id
        self.path = path
        self.speed = speed  # 每幀移動多少公尺，可再調整
        self.pos = path[0] if path else (0, 0)  # 當前位置
        self.heading = 0.0
        self.target_waypoint_index = 1
        self.done = False

    def update(self):
        """
        前進一個更新步驟，依據 self.speed 移動並更新船隻航向。

        移動方向根據從當前位置到下一個 waypoint 的向量計算，
        航向以 (270 - angle) % 360 來計算，使得圖片（預設船頭朝下）經過
        校正後，船頭正確顯示（此處調整後使船頭朝上）。
        """
        if self.done or len(self.path) <= 1:
            self.done = True
            return

        tx, ty = self.path[self.target_waypoint_index]
        cx, cy = self.pos
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)

        if dist < 1e-8:
            self.pos = (tx, ty)
            self.target_waypoint_index += 1
            if self.target_waypoint_index >= len(self.path):
                self.done = True
            return

        if dist <= self.speed:
            self.pos = (tx, ty)
            self.target_waypoint_index += 1
            if self.target_waypoint_index >= len(self.path):
                self.done = True
        else:
            ratio = self.speed / dist
            nx = cx + dx * ratio
            ny = cy + dy * ratio
            self.pos = (nx, ny)

        # 更新朝向：以從當前位置到目標 waypoint 的向量計算
        if dist != 0:
            angle_rad = math.atan2(dy, dx)
            # 計算公式：(270 - angle) % 360 使得水平方向的運動正確，
            # 並校正使船頭朝上（根據 ship.png 預設船頭朝下）
            self.heading = (270 - math.degrees(angle_rad)) % 360

    def current_line(self):
        """
        回傳尚未走完的路徑線段（僅繪製從當前位置到終點）。

        傳回:
          list: 由當前位置與剩餘 waypoint 組成的線段座標列表。
        """
        if self.done:
            return []
        # 取當前位置與剩下的 waypoint
        return [self.pos] + self.path[self.target_waypoint_index :]


def main():
    # -----------------------------------------------------------------------------
    # 定義多艘船（示範資料）
    # -----------------------------------------------------------------------------
    ships_data = [
{"id": "ShipA", "pos": (15.06, 3.45), "goal": (43.628, 21.234)},
{"id": "ShipB", "pos": (21.73, 3.45), "goal": (33.266, 17.614)},
{"id": "ShipC", "pos": (28.17, 3.45), "goal": (31.134, 32.528)},
{"id": "ShipD", "pos": (28.17, -3.22), "goal": (28.114, 23.166)},
{"id": "ShipE", "pos": (28.17, -9.66), "goal": (41.062, 30.762)},

    ]

    # -----------------------------------------------------------------------------
    # 平滑化參數預設與多船規劃結果計算
    # -----------------------------------------------------------------------------
    grid_scale = 0.5 # 網格大小（公尺）
    selected_method = "moving_average"  # 預設平滑化方法："none", "moving_average", 或 "bezier"

    # 透過 multi_ship_planning 產生結果（包含路徑、航向、分析資訊）
    results = multi_ship_planning(
        ships_data,
        grid_scale=grid_scale,
        smoothing_method=selected_method,
        safe_distance=5
    )

    # -----------------------------------------------------------------------------
    # Pygame 初始化與顯示範圍設定
    # -----------------------------------------------------------------------------
    pygame.init()
    screen_w, screen_h = 1200, 800
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("多船 FCC 路徑規劃 (Shift 切換模式)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    button_font = pygame.font.SysFont(None, 24)

    # 計算所有船路徑點以決定顯示範圍
    all_x = []
    all_y = []
    for sid, data in results.items():
        for px, py in data["path"]:
            all_x.append(px)
            all_y.append(py)
    if not all_x:
        all_x = [0, 1]
        all_y = [0, 1]
    margin_m = 3
    min_x = min(all_x) - margin_m
    max_x = max(all_x) + margin_m
    min_y = min(all_y) - margin_m
    max_y = max(all_y) + margin_m

    map_w = max_x - min_x
    map_h = max_y - min_y
    if map_w < 1e-6:
        map_w = 1
    if map_h < 1e-6:
        map_h = 1
    vis_scale = min(screen_w / map_w, screen_h / map_h)

    def trans(px, py):
        """
        將公尺座標轉換為螢幕座標。

        參數:
          px (float): X 座標（公尺）。
          py (float): Y 座標（公尺）。

        傳回:
          tuple: 螢幕座標 (sx, sy)。
        """
        sx = (px - min_x) * vis_scale
        sy = screen_h - ((py - min_y) * vis_scale)
        return sx, sy

    # -----------------------------------------------------------------------------
    # 為每艘船指定隨機顏色
    # -----------------------------------------------------------------------------
    color_map = {}
    for sid in results.keys():
        color_map[sid] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )

    # -----------------------------------------------------------------------------
    # 載入船圖 (預設圖片 ship.png 為船頭朝下)
    # -----------------------------------------------------------------------------
    ship_img_raw = pygame.image.load("ship.png").convert_alpha()
    ship_img_raw = pygame.transform.scale(ship_img_raw, (40, 40))

    # -----------------------------------------------------------------------------
    # 預先建立 AnimatedShip 物件（用於動畫模式）
    # -----------------------------------------------------------------------------
    animated_ships = {}
    for sid, data in results.items():
        path_m = data["path"]
        animated_ships[sid] = AnimatedShip(sid, path_m, speed=0.1)  # 每幀移動 0.1 公尺

    # -----------------------------------------------------------------------------
    # 建立平滑化方法按鈕區域（左上角）
    # -----------------------------------------------------------------------------
    none_btn = pygame.Rect(10, 50, 110, 30)
    mova_btn = pygame.Rect(10, 90, 110, 30)
    bezier_btn = pygame.Rect(10, 130, 110, 30)

    def draw_button(rect, text, is_active):
        """
        繪製按鈕，根據是否啟動選擇不同顏色。

        參數:
          rect (pygame.Rect): 按鈕區域。
          text (str): 按鈕文字。
          is_active (bool): 是否為當前選取的狀態。
        """
        color = (0, 200, 0) if is_active else (180, 180, 180)
        pygame.draw.rect(screen, color, rect, border_radius=5)
        txt_surf = button_font.render(text, True, (0, 0, 0))
        screen.blit(txt_surf, (rect.x + 5, rect.y + 5))

    def recalc_results(method):
        """
        根據指定的平滑化方法重新計算多船規劃結果。
        """
        return multi_ship_planning(
            ships_data, grid_scale=grid_scale, smoothing_method=method
        )

    # -----------------------------------------------------------------------------
    # 初始化模式狀態：動畫模式與暫停狀態
    # -----------------------------------------------------------------------------
    animation_mode = False
    paused = False

    running = True
    while running:
        dt = clock.tick(30)  # 每秒 30 幀
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (K_LSHIFT, K_RSHIFT):
                    # 按下 Shift 切換模式
                    animation_mode = not animation_mode
                    if animation_mode:
                        paused = False
                        # 進入動畫模式時，重置所有 AnimatedShip 物件
                        for sid in animated_ships:
                            s = animated_ships[sid]
                            s.pos = s.path[0] if s.path else (0, 0)
                            s.target_waypoint_index = 1
                            s.done = False
                    else:
                        # 回到步數模式，無特殊處理
                        pass
                elif event.key == pygame.K_SPACE:
                    # 按下空白鍵切換暫停狀態
                    paused = not paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 檢查按鈕點擊以選擇平滑化方法
                if none_btn.collidepoint(event.pos):
                    selected_method = "none"
                    results = recalc_results("none")
                    animated_ships = {}
                    for sid, data in results.items():
                        animated_ships[sid] = AnimatedShip(sid, data["path"], speed=0.1)
                elif mova_btn.collidepoint(event.pos):
                    selected_method = "moving_average"
                    results = recalc_results("moving_average")
                    animated_ships = {}
                    for sid, data in results.items():
                        animated_ships[sid] = AnimatedShip(sid, data["path"], speed=0.1)
                elif bezier_btn.collidepoint(event.pos):
                    selected_method = "bezier"
                    results = recalc_results("bezier")
                    animated_ships = {}
                    for sid, data in results.items():
                        animated_ships[sid] = AnimatedShip(sid, data["path"], speed=0.1)

        # -----------------------------------------------------------------------------
        # 塗背景與繪製網格
        # -----------------------------------------------------------------------------
        screen.fill((255, 255, 255))
        for gx in range(int(min_x), int(max_x) + 1):
            xx = (gx - min_x) * vis_scale
            pygame.draw.line(screen, (220, 220, 220), (xx, 0), (xx, screen_h))
        for gy in range(int(min_y), int(max_y) + 1):
            yy = screen_h - ((gy - min_y) * vis_scale)
            pygame.draw.line(screen, (220, 220, 220), (0, yy), (screen_w, yy))

        if not animation_mode:
            # -----------------------------------------------------------------------------
            # 步數模式：顯示每艘船的路徑、箭頭與步數
            # -----------------------------------------------------------------------------
            for sid, val in results.items():
                path = val["path"]
                heads = val["headings"]
                col = color_map[sid]
                for i, (px, py) in enumerate(path):
                    sx, sy = trans(px, py)
                    draw_arrow(screen, col, (sx, sy), heads[i], size=10)
                    step_txt = font.render(str(i), True, col)
                    screen.blit(step_txt, (sx + 2, sy + 2))
        else:
            # -----------------------------------------------------------------------------
            # 動畫模式
            # -----------------------------------------------------------------------------
            if not paused:
                for sid, ship_obj in animated_ships.items():
                    ship_obj.update()
            for sid, ship_obj in animated_ships.items():
                line_pts = ship_obj.current_line()
                col = color_map[sid]
                if len(line_pts) > 1:
                    screen_line = [trans(x, y) for (x, y) in line_pts]
                    pygame.draw.lines(screen, col, False, screen_line, 2)
                sx, sy = trans(ship_obj.pos[0], ship_obj.pos[1])
                rot_img = pygame.transform.rotate(ship_img_raw, -ship_obj.heading)
                w, h = rot_img.get_size()
                draw_x = sx - w / 2
                draw_y = sy - h / 2
                screen.blit(rot_img, (draw_x, draw_y))
            if paused:
                pause_txt = font.render("PAUSED", True, (255, 0, 0))
                screen.blit(pause_txt, (screen_w - 120, 10))

        # -----------------------------------------------------------------------------
        # 在左上角顯示模式提示文字
        # -----------------------------------------------------------------------------
        mode_str = "ANIMATION" if animation_mode else "STEP"
        hint_txt = font.render(
            f"Press [Shift] to Toggle Mode ({mode_str})   ", True, (0, 0, 0)
        )
        screen.blit(hint_txt, (5, 5))
        hint_txt = font.render("Press [Space] to pause/resume   ", True, (0, 0, 0))
        screen.blit(hint_txt, (5, 20))

        # -----------------------------------------------------------------------------
        # 畫按鈕：顯示平滑化方法選擇 (左上角)
        # -----------------------------------------------------------------------------
        draw_button(none_btn, "None", selected_method == "none")
        draw_button(mova_btn, "Moving", selected_method == "moving_average")
        draw_button(bezier_btn, "Bezier", selected_method == "bezier")

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
