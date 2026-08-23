import math
from pathlib import Path

import numpy as np

def load_important_cells(file_path: str):
    """
    Load important cells defined by AIV metric or eigencentrality metric
    """
    return np.load(file_path, allow_pickle=True).item()

def load_cell_ids_near_occlusion(file_path: str | Path):
    """
    Load cell ids near occlusion
    """
    return np.load(file_path, allow_pickle=False)



def is_cell_near_wall(cell_coordinates: tuple):
    if cell_coordinates is not None:
        if abs(cell_coordinates[1]) >= 8:
            return True
        else:
            # print(f"Condition 2: abs(cell_coordinates[1]) + abs(cell_coordinates[0]) ({abs(cell_coordinates[1]) + abs(cell_coordinates[0])}) >= 18 -> {abs(cell_coordinates[1]) + abs(cell_coordinates[0]) >= 18}")
            if abs(cell_coordinates[1]) + abs(cell_coordinates[0]) >= 18:
                return True
            else:
                return False
    else:
        return False



def find(cells: list, location: list):
    """
    Find the closest cell to the location
    Return the index of the closest cell
    """
    closest = -1
    closest_distance = 0
    for index in range(len(cells)):
        if cells[index] is not None:
            dist = math.dist(location, cells[index])
            if dist < closest_distance or closest == -1:
                closest = index
                closest_distance = dist
    return closest

def normalize_angle(angle_rad: float) -> float:
    """Normalizes an angle to the range [0, 2*pi)"""
    normalized_angle = angle_rad % (2 * math.pi)
    if normalized_angle < 0:
        normalized_angle += 2 * math.pi
    return normalized_angle

def find_aiv(aiv_list: list, current_direction_rad: float) -> float:
    """
    Find the AIV  metric for the closest cell,
    corresponding to the direction that matches the current agent's direction
    within a 60-degree range.

    Args:
        aiv_list (list): A list containing AIV metrics for different directions.
        current_direction_rad (float): The agent's current direction in radians,
                                    normalized to the range [0, 2*pi).

    """
    central_angles_deg_map = {0:60,
                              1:0,
                              2:-60,
                              3:-120,
                              4:-180,
                              5:-240}
    half_range_deg = 30

    for key, values in aiv_list.items():
        central_angle_rad = normalize_angle(math.radians(central_angles_deg_map[key]))

        angle_offset_rad = math.radians(half_range_deg)

        lower_bound_rad = central_angle_rad - angle_offset_rad
        upper_bound_rad = central_angle_rad + angle_offset_rad

        normalized_lower_bound = normalize_angle(lower_bound_rad)
        normalized_upper_bound = normalize_angle(upper_bound_rad)

        aiv_metric_value = values['added'] + values['removed']

        # Check if current_direction_rad is within the calculated range, handling wrap-around
        if normalized_lower_bound <= normalized_upper_bound:
            if normalized_lower_bound <= current_direction_rad <= normalized_upper_bound:
                return aiv_metric_value
        else: # Range wraps around 0/2pi (e.g., from 330 deg to 30 deg)
            if current_direction_rad >= normalized_lower_bound or current_direction_rad <= normalized_upper_bound:
                return aiv_metric_value

    # If no matching direction range is found, return 0 as AIV
    return 0
