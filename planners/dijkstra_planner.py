import cv2
import numpy as np
from typing import List, Tuple
import math
import heapq
from scipy import interpolate

def create_planner(field_width: float, field_height: float, step: float,
                   robot_radius: float, obstacle_safety: float,
                   edge_limit_cm: float, refine_step: float = 1.0) -> dict:
    grid_width = int(field_width / step) + 1
    grid_height = int(field_height / step) + 1

    planner = {
        'field_width': field_width,
        'field_height': field_height,
        'step': step,
        'robot_radius': robot_radius,
        'obstacle_safety': obstacle_safety,
        'edge_limit_cm': edge_limit_cm,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'obstacles': [],
        'obstacle_map': np.zeros((grid_height, grid_width), dtype=np.uint8),
        'path': [],
        'spline_path': [],
        'refined_path': [],
        'path_locked': False,
        'current_goal': None,
        'fine_step': refine_step
    }
    return planner

def update_obstacles(planner: dict, obstacles: List[dict]):
    planner['obstacles'] = obstacles
    if planner.get('path_locked', False):
        return
    planner['obstacle_map'].fill(0)

    step = planner['step']
    grid_width = planner['grid_width']
    grid_height = planner['grid_height']
    field_width = planner['field_width']
    field_height = planner['field_height']
    rectified_width = 720
    rectified_height = 720

    for obs in obstacles:
        if 'expanded_contour' in obs:
            expanded_contour = obs['expanded_contour']

            contour_real = []
            for point in expanded_contour:
                if len(point) == 2:
                    px, py = point
                else:
                    px, py = point[0]

                real_x = px * (field_width / rectified_width)
                real_y = (rectified_height - py) * (field_height / rectified_height)
                contour_real.append([real_x, real_y])

            if len(contour_real) >= 3:
                grid_points = []
                for point in contour_real:
                    gx = int(point[0] / step)
                    gy = int(point[1] / step)
                    if 0 <= gx < grid_width and 0 <= gy < grid_height:
                        grid_points.append([gx, gy])

                if len(grid_points) >= 3:
                    grid_points = np.array(grid_points, dtype=np.int32)
                    cv2.fillPoly(planner['obstacle_map'], [grid_points], 1)


def reset_path(planner: dict):
    planner['path'] = []
    planner['spline_path'] = []
    planner['refined_path'] = []
    planner['path_locked'] = False
    planner['current_goal'] = None

def is_cell_safe(planner: dict, grid_x: int, grid_y: int) -> bool:
    if not (0 <= grid_x < planner['grid_width'] and 0 <= grid_y < planner['grid_height']):
        return False

    if planner['obstacle_map'][grid_y, grid_x] == 1:
        return False

    cx = (grid_x + 0.5) * planner['step']
    cy = (grid_y + 0.5) * planner['step']

    if (cx < planner['edge_limit_cm'] or
            cx > planner['field_width'] - planner['edge_limit_cm'] or
            cy < planner['edge_limit_cm'] or
            cy > planner['field_height'] - planner['edge_limit_cm']):
        return False
    return True

def get_step_cost(dx: int, dy: int, step: float) -> float:
    if dx != 0 and dy != 0:
        return math.sqrt(2) * step
    return step

def world_to_grid(planner: dict, x: float, y: float) -> Tuple[int, int]:
    step = planner['step']
    grid_x = int(x / step)
    grid_y = int(y / step)
    grid_x = max(0, min(grid_x, planner['grid_width'] - 1))
    grid_y = max(0, min(grid_y, planner['grid_height'] - 1))
    return grid_x, grid_y


def grid_to_world(planner: dict, grid_x: int, grid_y: int) -> Tuple[float, float]:
    step = planner['step']
    return (grid_x + 0.5) * step, (grid_y + 0.5) * step


def build_spline_from_path(path: List[Tuple[float, float]], planner: dict = None, num_points: int = 200) -> List[
    Tuple[float, float]]:
    """
    Строит плавный сплайн по точкам пути Дейкстры (сглаживает, а не проходит через все точки)
    """
    if len(path) < 3:
        return path.copy()

    # Разделяем координаты
    points = np.array(path)
    x = points[:, 0]
    y = points[:, 1]

    # Вычисляем параметр t (длина дуги для более равномерного распределения)
    t = np.zeros(len(points))
    for i in range(1, len(points)):
        dist = math.hypot(points[i, 0] - points[i - 1, 0], points[i, 1] - points[i - 1, 1])
        t[i] = t[i - 1] + dist

    # Нормализуем t от 0 до 1
    if t[-1] > 0:
        t = t / t[-1]
    else:
        return path.copy()

    try:
        # Убираем дубликаты по t если они есть
        unique_indices = []
        for i in range(len(t)):
            if i == 0 or t[i] > t[i - 1] + 1e-6:
                unique_indices.append(i)

        t_unique = t[unique_indices]
        x_unique = x[unique_indices]
        y_unique = y[unique_indices]

        if len(t_unique) < 3:
            return path.copy()

        # Используем UnivariateSpline с параметром сглаживания s
        # Чем больше s, тем сильнее сглаживание (и тем больше сплайн отклоняется от точек)
        from scipy.interpolate import UnivariateSpline

        # Параметр сглаживания:
        # - s = 0: проходит через все точки (как сейчас)
        # - s = len(x): умеренное сглаживание
        # - s = len(x) * 10: сильное сглаживание
        smoothing_factor = len(x_unique) * 5  # Умеренное сглаживание

        fx = UnivariateSpline(t_unique, x_unique, s=smoothing_factor)
        fy = UnivariateSpline(t_unique, y_unique, s=smoothing_factor)

        # Генерируем точки
        t_new = np.linspace(0, 1, num_points)

        x_spline = fx(t_new)
        y_spline = fy(t_new)

        # Собираем точки сплайна
        spline_path = list(zip(x_spline, y_spline))

        # Проверяем безопасность (чтобы сплайн не врезался в препятствия)
        if planner is not None:
            corrected_path = []
            step = planner['step']

            for px, py in spline_path:
                grid_x, grid_y = world_to_grid(planner, px, py)
                if is_cell_safe(planner, grid_x, grid_y):
                    corrected_path.append((px, py))
                else:
                    # Если точка в препятствии, тянем её к ближайшей безопасной
                    found = False
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            test_x = px + dx
                            test_y = py + dy
                            gx, gy = world_to_grid(planner, test_x, test_y)
                            if is_cell_safe(planner, gx, gy):
                                corrected_path.append((test_x, test_y))
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        corrected_path.append((px, py))

            spline_path = corrected_path

        return spline_path

    except Exception as e:
        print(f"Ошибка при построении сплайна: {e}")
        return path.copy()

def refine_spline_to_grid(spline_path: List[Tuple[float, float]], planner: dict) -> List[Tuple[float, float]]:
    if len(spline_path) < 2:
        return spline_path.copy()

    fine_step = planner['fine_step']
    refined_path = []

    # Проходим по сплайну и добавляем точки с мелким шагом
    for i in range(len(spline_path) - 1):
        x1, y1 = spline_path[i]
        x2, y2 = spline_path[i + 1]

        # Если это первая точка
        if i == 0:
            refined_path.append((x1, y1))

        # Вычисляем расстояние между точками сплайна
        dist = math.hypot(x2 - x1, y2 - y1)

        # Если расстояние больше мелкого шага, добавляем промежуточные точки
        if dist > fine_step:
            num_intermediate = int(dist / fine_step)
            for j in range(1, num_intermediate + 1):
                t = j / (num_intermediate + 1)
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                refined_path.append((ix, iy))
        else:
            # Добавляем точку если расстояние меньше шага
            if i < len(spline_path) - 1:
                refined_path.append((x2, y2))

    # Добавляем последнюю точку
    if len(refined_path) == 0 or refined_path[-1] != spline_path[-1]:
        refined_path.append(spline_path[-1])

    return refined_path

def find_path(planner: dict, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
    if planner.get('path_locked', False) and planner.get('current_goal') == goal:
        return planner['path']

    planner['path_locked'] = False
    start_grid = world_to_grid(planner, start[0], start[1])
    goal_grid = world_to_grid(planner, goal[0], goal[1])

    if not is_cell_safe(planner, start_grid[0], start_grid[1]):
        print(" Стартовая точка небезопасна")
        return []

    if not is_cell_safe(planner, goal_grid[0], goal_grid[1]):
        print(" Целевая точка небезопасна")
        return []

    moves = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]

    INF = float('inf')
    dist = {}
    parent = {}

    start_node = (start_grid[0], start_grid[1])
    dist[start_node] = 0
    pq = [(0, start_grid[0], start_grid[1])]

    while pq:
        current_dist, x, y = heapq.heappop(pq)
        current = (x, y)

        if current == (goal_grid[0], goal_grid[1]):
            path = []
            curr = current
            while curr in parent:
                path.append(grid_to_world(planner, curr[0], curr[1]))
                curr = parent[curr]
            path.append(grid_to_world(planner, start_grid[0], start_grid[1]))
            path.reverse()

            # Строим плавный сплайн по точкам Дейкстры
            spline_path = build_spline_from_path(path, planner, num_points=250)

            # Аппроксимируем сплайн точками с мелким шагом
            refined_path = refine_spline_to_grid(spline_path, planner)

            # Сохраняем все версии пути
            planner['path'] = path
            planner['spline_path'] = spline_path
            planner['refined_path'] = refined_path
            planner['path_locked'] = True
            planner['current_goal'] = goal

            return path

        if current_dist > dist.get(current, INF):
            continue

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if not (0 <= nx < planner['grid_width'] and 0 <= ny < planner['grid_height']):
                continue
            if not is_cell_safe(planner, nx, ny):
                continue

            step_cost = get_step_cost(dx, dy, planner['step'])
            new_dist = current_dist + step_cost

            if new_dist < dist.get(neighbor, INF):
                dist[neighbor] = new_dist
                parent[neighbor] = current
                heapq.heappush(pq, (new_dist, nx, ny))

    print(" Путь не найден")
    return []

def get_velocities(planner: dict, current_x: float, current_y: float,
                   max_speed: float, kp: float, acc_speed_error: float) -> Tuple[float, float]:
    # Используем аппроксимированный сплайн путь для более плавного движения
    path = planner['refined_path'] if planner['refined_path'] else planner['path']

    if not path or len(path) < 2:
        return 0.0, 0.0

    min_dist = float('inf')
    nearest_idx = 0
    for i, point in enumerate(path):
        px, py = point
        dist = math.hypot(px - current_x, py - current_y)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    target_idx = min(nearest_idx + 5, len(path) - 1)
    target_x, target_y = path[target_idx]

    error_x = target_x - current_x
    error_y = target_y - current_y
    error_distance = math.hypot(error_x, error_y)

    min_speed_ms = 0.03

    max_speed_cm = max_speed * 100.0
    speed_cm = min(kp * error_distance, max_speed_cm)

    final_goal = path[-1]
    dist_to_final = math.hypot(final_goal[0] - current_x, final_goal[1] - current_y)
    if dist_to_final > acc_speed_error:
        speed_cm = max(speed_cm, min_speed_ms * 100.0)

    if error_distance > 0:
        vx = (error_x / error_distance) * (speed_cm / 100.0)
        vy = (error_y / error_distance) * (speed_cm / 100.0)
    else:
        vx, vy = 0.0, 0.0

    return vx, -vy

def draw_planning_contours(planner: dict, frame: np.ndarray) -> np.ndarray:
    for obs in planner['obstacles']:
        if 'expanded_contour' in obs:
            expanded_contour = obs['expanded_contour']
            if len(expanded_contour) > 2:
                cv2.polylines(frame, [expanded_contour], True, (255, 0, 0), 2)
    return frame


def draw_path_on_frame(planner: dict, frame: np.ndarray, path: List[Tuple[float, float]],
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Рисует все три версии пути на кадре:
    - Синий: оригинальный путь Дейкстры (толщина 3)
    - Зеленый: сплайн (толщина 2)
    - Красный: аппроксимированный путь (толщина 1)
    """
    if not path or len(path) < 2:
        return frame

    h, w = frame.shape[:2]
    field_width = planner['field_width']
    field_height = planner['field_height']

    # 1. Рисуем оригинальный путь Дейкстры (синий, толщина 3)
    if planner['path'] and len(planner['path']) > 1:
        points_dijkstra = []
        for real_x, real_y in planner['path']:
            x_px = int(real_x / field_width * w)
            y_px = int(h - (real_y / field_height * h))
            points_dijkstra.append((x_px, y_px))

        for i in range(len(points_dijkstra) - 1):
            cv2.line(frame, points_dijkstra[i], points_dijkstra[i + 1], (255, 0, 0), 3)

    # 3. Рисуем аппроксимированный путь (красный, толщина 1)
    if planner['refined_path'] and len(planner['refined_path']) > 1:
        points_refined = []
        for real_x, real_y in planner['refined_path']:
            x_px = int(real_x / field_width * w)
            y_px = int(h - (real_y / field_height * h))
            points_refined.append((x_px, y_px))

        for i in range(len(points_refined) - 1):
            cv2.line(frame, points_refined[i], points_refined[i + 1], (0, 0, 255), 2)

    # Добавляем легенду
    legend_y = h - 20
    cv2.putText(frame, "Blue: Dijkstra (thick)", (10, legend_y - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(frame, "Green: Spline (medium)", (10, legend_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "Red: Refined (thin)", (10, legend_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame