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
from shapely.ops import nearest_points, unary_union

from .contracts import validate_xy


DEFAULT_PROJECTION_EPSILON = 1e-5
DESTINATION_MODES = ("project", "reject")
NAVIGATION_CONTRACT = "graph_route_exact_xy_terminal_project_v2"


def navigation_contract(destination_mode: str) -> str:
    if destination_mode not in DESTINATION_MODES:
        raise ValueError("destination_mode must be project or reject")
    return f"graph_route_exact_xy_terminal_{destination_mode}_v2"


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

    def __init__(self, navigation, *, arena, occlusions, destination_mode: str = "project",
                 projection_epsilon: float = DEFAULT_PROJECTION_EPSILON):
        self.contract = navigation_contract(destination_mode)
        if (isinstance(projection_epsilon, bool) or not math.isfinite(projection_epsilon)
                or not 0.0 < projection_epsilon <= 0.01):
            raise ValueError("projection_epsilon must be finite and in (0, 0.01]")
        self.destination_mode = destination_mode
        self.projection_epsilon = float(projection_epsilon)
        self.base = navigation
        arena_polygon = _polygon(arena)
        obstacles = [_polygon(value) for value in occlusions]
        self.free_space = arena_polygon.difference(unary_union(obstacles)) if obstacles else arena_polygon
        if self.free_space.is_empty or not self.free_space.is_valid:
            raise ValueError("Exact XY navigation requires valid static free-space geometry")
        self.locations = tuple(tuple(map(float, point)) for point in navigation.points.point_list if point is not None)
        self.last_invalid_reason: str | None = None
        self.components = tuple(self._polygons(self.free_space))
        self.inset_components = tuple(component.buffer(-self.projection_epsilon)
                                      for component in self.components)

    @staticmethod
    def _polygons(geometry):
        if geometry.geom_type == "Polygon":
            yield geometry
        elif hasattr(geometry, "geoms"):
            for child in geometry.geoms:
                yield from ExactXYNavigation._polygons(child)

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

    def resolve_destination(self, src, dst) -> dict[str, Any]:
        """Keep feasible requests unchanged, otherwise project or explicitly hold.

        Projection uses only static free-space geometry and the public current
        prey position. The small inset is numerical clearance, not a promise
        that the finite-size body can traverse a route; simulator collision
        handling remains authoritative. A failed projection never becomes an
        unreported nearest-cell command.
        """
        source, requested = validate_xy(src), validate_xy(dst)
        self.get_path(source, requested)
        original_reason = self.last_invalid_reason
        result = {
            "requested_xy": list(requested), "executed_xy": list(requested),
            "requested_invalid_destination": original_reason is not None,
            "original_invalid_reason": original_reason,
            "was_projected": False, "projection_distance": 0.0,
            "projection_attempted": False,
            "invalid_destination": original_reason is not None,
            "invalid_destination_reason": original_reason,
        }
        if original_reason is None:
            return result
        result["executed_xy"] = list(source)
        if self.destination_mode == "reject":
            return result
        result["projection_attempted"] = True
        source_point = Point(source)
        index = next((i for i, component in enumerate(self.components)
                      if component.covers(source_point)), None)
        if index is None:
            self.last_invalid_reason = "prey_outside_free_space_component"
        elif self.inset_components[index].is_empty:
            self.last_invalid_reason = "current_free_space_component_has_empty_inset"
        else:
            target_point = nearest_points(self.inset_components[index], Point(requested))[0]
            projected = validate_xy((target_point.x, target_point.y))
            self.get_path(source, projected)
            if self.last_invalid_reason is None:
                result.update({
                    "executed_xy": list(projected), "was_projected": True,
                    "projection_distance": math.dist(requested, projected),
                    "invalid_destination": False, "invalid_destination_reason": None,
                })
                return result
        result["invalid_destination_reason"] = self.last_invalid_reason
        return result


def install_exact_xy_navigation(model, *, destination_mode: str = "project",
                                projection_epsilon: float = DEFAULT_PROJECTION_EPSILON) -> ExactXYNavigation:
    """Change only prey navigation; shared loader/predator routing stays intact."""
    current = model.prey.navigation
    if isinstance(current, ExactXYNavigation):
        if current.destination_mode == destination_mode and current.projection_epsilon == projection_epsilon:
            return current
        current = current.base
    adapter = ExactXYNavigation(current, arena=model.arena, occlusions=model.occlusions,
                                destination_mode=destination_mode, projection_epsilon=projection_epsilon)
    model.prey.navigation = adapter
    return adapter
