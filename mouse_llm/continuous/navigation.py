"""Task-local exact terminal waypoints over Cellworld's graph router.

The vendored router maps every destination to a cell center. This adapter
keeps graph routing, but retains a non-center requested XY as the final
waypoint. The simulator still integrates motion and resolves body collisions.
Only static map geometry is used here; predator state is never consulted.
"""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .contracts import validate_xy


NAVIGATION_CONTRACT = "graph_route_exact_xy_terminal_v1"


def _polygon(value: Any) -> Polygon:
    if isinstance(value, Polygon):
        return value
    vertices = value.vertices
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().tolist()
    return Polygon(vertices)


class ExactXYNavigation:
    """Preserve valid XY endpoints, and explicitly reject unroutable requests."""

    contract = NAVIGATION_CONTRACT

    def __init__(self, navigation, *, arena, occlusions):
        self.base = navigation
        arena_polygon = _polygon(arena)
        obstacles = [_polygon(value) for value in occlusions]
        self.free_space = arena_polygon.difference(unary_union(obstacles)) if obstacles else arena_polygon
        if self.free_space.is_empty or not self.free_space.is_valid:
            raise ValueError("Exact XY navigation requires valid static free-space geometry")
        self.locations = tuple(tuple(map(float, point)) for point in navigation.points.point_list if point is not None)
        self.last_invalid_reason: str | None = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def valid_destination(self, destination) -> bool:
        return bool(self.free_space.contains(Point(validate_xy(destination))))

    def _clear(self, source, destination) -> bool:
        if source == destination:
            return bool(self.free_space.covers(Point(source)))
        return bool(self.free_space.covers(LineString([source, destination])))

    def _route_clear(self, source, path) -> bool:
        for destination in path:
            if not self._clear(source, destination):
                return False
            source = destination
        return True

    def get_path(self, src, dst):
        source, destination = validate_xy(src), validate_xy(dst)
        self.last_invalid_reason = None
        if not self.valid_destination(destination):
            self.last_invalid_reason = "destination_outside_free_space"
            return []
        if math.dist(source, destination) <= 1e-12:
            return []
        # Keep the existing reference's routing for exact catalog destinations.
        is_center = any(math.dist(destination, point) <= 1e-10 for point in self.locations)
        if not is_center and self._clear(source, destination):
            return [destination]
        native_path = [tuple(map(float, point)) for point in self.base.get_path(src=source, dst=destination)]
        if is_center and native_path and self._route_clear(source, native_path):
            return native_path
        if (native_path and self._route_clear(source, native_path)
                and self._clear(native_path[-1], destination)):
            return native_path if native_path[-1] == destination else [*native_path, destination]
        # A nearest cell need not see an off-center destination across an edge.
        # Try visible graph anchors, preserving the requested endpoint exactly.
        for anchor in sorted(self.locations, key=lambda point: math.dist(point, destination)):
            if not self._clear(anchor, destination):
                continue
            route = [tuple(map(float, point)) for point in self.base.get_path(src=source, dst=anchor)]
            if route and self._route_clear(source, route):
                return route if route[-1] == destination else [*route, destination]
        self.last_invalid_reason = "no_clear_graph_route_to_requested_destination"
        return []


def install_exact_xy_navigation(model) -> ExactXYNavigation:
    """Change only prey navigation; shared loader/predator routing stays intact."""
    current = model.prey.navigation
    if isinstance(current, ExactXYNavigation):
        return current
    adapter = ExactXYNavigation(current, arena=model.arena, occlusions=model.occlusions)
    model.prey.navigation = adapter
    return adapter
