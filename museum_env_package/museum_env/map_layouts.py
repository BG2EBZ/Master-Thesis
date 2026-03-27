from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

GEOMETRY_EPS = 1e-5


@dataclass(frozen=True)
class AxisAlignedRect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def apply_margin(self, margin: float, *, expand_degenerate: bool = False) -> "AxisAlignedRect":
        """Shrink normal axes by margin and optionally expand degenerate doorway axes."""
        m = max(0.0, float(margin))
        if self.xmin < self.xmax:
            xmin = self.xmin + m
            xmax = self.xmax - m
        elif expand_degenerate:
            xmin = self.xmin - m
            xmax = self.xmax + m
        else:
            xmin = self.xmin + m
            xmax = self.xmax - m

        if self.ymin < self.ymax:
            ymin = self.ymin + m
            ymax = self.ymax - m
        elif expand_degenerate:
            ymin = self.ymin - m
            ymax = self.ymax + m
        else:
            ymin = self.ymin + m
            ymax = self.ymax - m
        return AxisAlignedRect(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def is_valid(self) -> bool:
        return bool(self.xmin <= self.xmax and self.ymin <= self.ymax)

    def center(self) -> np.ndarray:
        return np.array(
            [0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax)],
            dtype=np.float32,
        )

    def area(self) -> float:
        return max(0.0, float(self.xmax - self.xmin)) * max(0.0, float(self.ymax - self.ymin))

    def contains_point(self, xy) -> bool:
        x = float(xy[0])
        y = float(xy[1])
        return bool(self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax)

    def project_point(self, xy) -> np.ndarray:
        point = np.array(xy, dtype=np.float32)
        return np.array(
            [
                float(np.clip(point[0], self.xmin, self.xmax)),
                float(np.clip(point[1], self.ymin, self.ymax)),
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class MapLayout:
    name: str
    default_xml_asset: str
    walkable_rects: tuple[AxisAlignedRect, ...]
    spawn_rects: tuple[AxisAlignedRect, ...]
    robot_waypoints: tuple[tuple[float, float], ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "_walkable_rect_cache", {})
        object.__setattr__(self, "_spawn_rect_cache", {})

    def _margin_rects(self, rects, margin: float, *, expand_degenerate: bool) -> tuple[AxisAlignedRect, ...]:
        valid_rects = []
        for rect in rects:
            candidate = rect.apply_margin(margin, expand_degenerate=expand_degenerate)
            if candidate.is_valid():
                valid_rects.append(candidate)
        return tuple(valid_rects)

    def _get_cached_margin_rects(self, cache_name: str, rects, margin: float, *, expand_degenerate: bool):
        margin_key = float(margin)
        cache = getattr(self, cache_name)
        cached_rects = cache.get(margin_key)
        if cached_rects is None:
            cached_rects = self._margin_rects(rects, margin_key, expand_degenerate=expand_degenerate)
            cache[margin_key] = cached_rects
        return cached_rects

    def get_walkable_rects(self, margin: float) -> tuple[AxisAlignedRect, ...]:
        return self._get_cached_margin_rects(
            "_walkable_rect_cache",
            self.walkable_rects,
            margin,
            expand_degenerate=True,
        )

    def get_spawn_rects(self, margin: float) -> tuple[AxisAlignedRect, ...]:
        return self._get_cached_margin_rects(
            "_spawn_rect_cache",
            self.spawn_rects,
            margin,
            expand_degenerate=False,
        )

    @staticmethod
    def _sample_point_in_rects(rects, rng=None, fallback_xy=None) -> np.ndarray:
        if not rects:
            if fallback_xy is None:
                return np.array([0.0, 0.0], dtype=np.float32)
            return np.array(fallback_xy, dtype=np.float32)

        rng_choice = np.random if rng is None else rng
        areas = np.array([rect.area() for rect in rects], dtype=np.float64)
        total_area = float(np.sum(areas))
        if total_area <= 0.0:
            return rects[0].center()

        probs = areas / total_area
        rect_idx = int(rng_choice.choice(len(rects), p=probs))
        rect = rects[rect_idx]
        return np.array(
            [
                float(rng_choice.uniform(rect.xmin, rect.xmax)),
                float(rng_choice.uniform(rect.ymin, rect.ymax)),
            ],
            dtype=np.float32,
        )

    def sample_walkable_point(self, margin: float, rng=None) -> np.ndarray:
        rects = self.get_walkable_rects(margin)
        return self._sample_point_in_rects(rects, rng=rng)

    def sample_spawn_point(self, margin: float, rng=None) -> np.ndarray:
        rects = self.get_spawn_rects(margin)
        fallback_xy = self.spawn_rects[0].center() if self.spawn_rects else np.array([0.0, 0.0], dtype=np.float32)
        return self._sample_point_in_rects(rects, rng=rng, fallback_xy=fallback_xy)

    @staticmethod
    def _contains_point_in_rects(x: float, y: float, rects) -> bool:
        return any(rect.xmin <= x <= rect.xmax and rect.ymin <= y <= rect.ymax for rect in rects)

    def contains_point(self, xy, margin: float) -> bool:
        x = float(xy[0])
        y = float(xy[1])
        rects = self.get_walkable_rects(margin)
        return self._contains_point_in_rects(x, y, rects)

    def _project_point_to_rects(self, xy, rects) -> np.ndarray:
        point = np.array(xy, dtype=np.float32)
        if not rects:
            return point

        point_x = float(point[0])
        point_y = float(point[1])
        best_projection = point
        best_dist_sq = None
        for rect in rects:
            if rect.xmin <= point_x <= rect.xmax and rect.ymin <= point_y <= rect.ymax:
                return point

            projected_x = float(np.clip(point_x, rect.xmin, rect.xmax))
            projected_y = float(np.clip(point_y, rect.ymin, rect.ymax))
            dx = projected_x - point_x
            dy = projected_y - point_y
            dist_sq = (dx * dx) + (dy * dy)
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_projection = np.array([projected_x, projected_y], dtype=np.float32)
        return best_projection

    def project_point(self, xy, margin: float) -> np.ndarray:
        rects = self.get_walkable_rects(margin)
        return self._project_point_to_rects(xy, rects)

    @staticmethod
    def _segment_rect_interval(start_xy, end_xy, rect: AxisAlignedRect):
        start_x = float(start_xy[0])
        start_y = float(start_xy[1])
        end_x = float(end_xy[0])
        end_y = float(end_xy[1])
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        if (delta_x * delta_x) + (delta_y * delta_y) <= (GEOMETRY_EPS * GEOMETRY_EPS):
            if rect.xmin <= start_x <= rect.xmax and rect.ymin <= start_y <= rect.ymax:
                return (0.0, 1.0)
            return None

        t_enter = 0.0
        t_exit = 1.0
        for coord, direction, lower, upper in (
            (start_x, delta_x, float(rect.xmin), float(rect.xmax)),
            (start_y, delta_y, float(rect.ymin), float(rect.ymax)),
        ):
            if abs(direction) <= GEOMETRY_EPS:
                if coord < lower - GEOMETRY_EPS or coord > upper + GEOMETRY_EPS:
                    return None
                continue

            inv_direction = 1.0 / direction
            axis_t0 = (lower - coord) * inv_direction
            axis_t1 = (upper - coord) * inv_direction
            if axis_t0 > axis_t1:
                axis_t0, axis_t1 = axis_t1, axis_t0

            t_enter = max(t_enter, axis_t0)
            t_exit = min(t_exit, axis_t1)
            if t_enter > t_exit + GEOMETRY_EPS:
                return None

        if t_exit < -GEOMETRY_EPS or t_enter > 1.0 + GEOMETRY_EPS:
            return None
        return (max(0.0, t_enter), min(1.0, t_exit))

    def _merged_walkable_intervals_in_rects(self, start_xy, end_xy, rects) -> tuple[tuple[float, float], ...]:
        intervals = []
        for rect in rects:
            interval = self._segment_rect_interval(start_xy, end_xy, rect)
            if interval is not None:
                intervals.append(interval)

        if not intervals:
            return ()

        intervals.sort(key=lambda interval: interval[0])
        merged = [intervals[0]]
        for interval_start, interval_end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if interval_start <= prev_end + GEOMETRY_EPS:
                merged[-1] = (prev_start, max(prev_end, interval_end))
            else:
                merged.append((interval_start, interval_end))
        return tuple(merged)

    def _merged_walkable_intervals(self, start_xy, end_xy, margin: float) -> tuple[tuple[float, float], ...]:
        rects = self.get_walkable_rects(margin)
        return self._merged_walkable_intervals_in_rects(start_xy, end_xy, rects)

    def _is_segment_walkable_in_rects(self, start_xy, end_xy, rects) -> bool:
        start_xy = np.array(start_xy, dtype=np.float32)
        end_xy = np.array(end_xy, dtype=np.float32)
        if not rects:
            return False

        if not self._contains_point_in_rects(float(start_xy[0]), float(start_xy[1]), rects):
            return False
        if not self._contains_point_in_rects(float(end_xy[0]), float(end_xy[1]), rects):
            return False

        merged = self._merged_walkable_intervals_in_rects(start_xy, end_xy, rects)
        if len(merged) != 1:
            return False
        return bool(merged[0][0] <= GEOMETRY_EPS and merged[0][1] >= 1.0 - GEOMETRY_EPS)

    def is_segment_walkable(self, start_xy, end_xy, margin: float) -> bool:
        rects = self.get_walkable_rects(margin)
        return self._is_segment_walkable_in_rects(start_xy, end_xy, rects)

    def _find_farthest_walkable_point_on_segment_in_rects(self, start_xy, end_xy, rects) -> np.ndarray:
        start_xy = np.array(start_xy, dtype=np.float32)
        end_xy = np.array(end_xy, dtype=np.float32)
        if not rects:
            return start_xy
        if not self._contains_point_in_rects(float(start_xy[0]), float(start_xy[1]), rects):
            return self._project_point_to_rects(start_xy, rects)

        merged = self._merged_walkable_intervals_in_rects(start_xy, end_xy, rects)
        if not merged:
            return start_xy

        farthest_t = 0.0
        for interval_start, interval_end in merged:
            if interval_start <= GEOMETRY_EPS <= interval_end + GEOMETRY_EPS:
                farthest_t = min(1.0, interval_end)
                break

        if farthest_t >= 1.0 - GEOMETRY_EPS:
            return end_xy
        return np.array(start_xy + farthest_t * (end_xy - start_xy), dtype=np.float32)

    def find_farthest_walkable_point_on_segment(self, start_xy, end_xy, margin: float) -> np.ndarray:
        rects = self.get_walkable_rects(margin)
        return self._find_farthest_walkable_point_on_segment_in_rects(start_xy, end_xy, rects)


DEFAULT_MUSEUM_LAYOUT = MapLayout(
    name="museum_default",
    default_xml_asset="museum_scene.xml",
    walkable_rects=(
        AxisAlignedRect(0.0, 10.0, 0.0, 10.0),
        AxisAlignedRect(7.0, 10.0, -10.0, 0.0),
        AxisAlignedRect(7.0, 12.0, -15.0, -10.0),
        AxisAlignedRect(7.0, 10.0, 0.0, 0.0),
        AxisAlignedRect(7.0, 10.0, -10.0, -10.0),
    ),
    spawn_rects=(
        AxisAlignedRect(0.0, 10.0, 0.0, 10.0),
    ),
    robot_waypoints=(
        (1.0, 5.0),
        (0.6, 4.5),
        (1.0, 2.0),
        (8.5, 2.0),
        (8.5, -10.0),
        (8.5, -12.5),
        (11.0, -12.5),
    ),
    metadata={
        "rooms": ("room_a", "corridor", "room_b"),
    },
)


MAP_LAYOUT_REGISTRY = {
    DEFAULT_MUSEUM_LAYOUT.name: DEFAULT_MUSEUM_LAYOUT,
}


def get_map_layout(name: str) -> MapLayout:
    try:
        return MAP_LAYOUT_REGISTRY[str(name)]
    except KeyError as exc:
        available = ", ".join(sorted(MAP_LAYOUT_REGISTRY.keys()))
        raise ValueError(f"Unknown map layout '{name}'. Available layouts: {available}") from exc
