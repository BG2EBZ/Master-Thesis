from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

DEFAULT_SEGMENT_CHECK_SPACING = 0.05
DEFAULT_BINARY_SEARCH_ITERS = 5


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
    segment_check_spacing: float = DEFAULT_SEGMENT_CHECK_SPACING
    binary_search_iters: int = DEFAULT_BINARY_SEARCH_ITERS

    def _margin_rects(self, rects, margin: float, *, expand_degenerate: bool) -> tuple[AxisAlignedRect, ...]:
        valid_rects = []
        for rect in rects:
            candidate = rect.apply_margin(margin, expand_degenerate=expand_degenerate)
            if candidate.is_valid():
                valid_rects.append(candidate)
        return tuple(valid_rects)

    def get_walkable_rects(self, margin: float) -> tuple[AxisAlignedRect, ...]:
        return self._margin_rects(self.walkable_rects, margin, expand_degenerate=True)

    def get_spawn_rects(self, margin: float) -> tuple[AxisAlignedRect, ...]:
        return self._margin_rects(self.spawn_rects, margin, expand_degenerate=False)

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

    def contains_point(self, xy, margin: float) -> bool:
        return any(rect.contains_point(xy) for rect in self.get_walkable_rects(margin))

    def project_point(self, xy, margin: float) -> np.ndarray:
        point = np.array(xy, dtype=np.float32)
        rects = self.get_walkable_rects(margin)
        if not rects or self.contains_point(point, margin):
            return point

        best_projection = None
        best_dist_sq = None
        for rect in rects:
            projected = rect.project_point(point)
            dist_sq = float(np.sum((projected - point) ** 2))
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_projection = projected
        if best_projection is None:
            return point
        return best_projection

    def is_segment_walkable(self, start_xy, end_xy, margin: float) -> bool:
        start_xy = np.array(start_xy, dtype=np.float32)
        end_xy = np.array(end_xy, dtype=np.float32)
        if not self.contains_point(start_xy, margin):
            return False
        if not self.contains_point(end_xy, margin):
            return False

        segment = end_xy - start_xy
        dist = float(np.linalg.norm(segment))
        if dist <= 1e-6:
            return True

        n_steps = max(1, int(np.ceil(dist / float(self.segment_check_spacing))))
        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32):
            point = start_xy + alpha * segment
            if not self.contains_point(point, margin):
                return False
        return True

    def find_farthest_walkable_point_on_segment(self, start_xy, end_xy, margin: float) -> np.ndarray:
        start_xy = np.array(start_xy, dtype=np.float32)
        end_xy = np.array(end_xy, dtype=np.float32)
        if not self.contains_point(start_xy, margin):
            return self.project_point(start_xy, margin)
        if self.is_segment_walkable(start_xy, end_xy, margin):
            return end_xy

        best_point = start_xy.copy()
        lo = 0.0
        hi = 1.0
        for _ in range(int(self.binary_search_iters)):
            mid = 0.5 * (lo + hi)
            candidate = start_xy + mid * (end_xy - start_xy)
            if self.is_segment_walkable(start_xy, candidate, margin):
                best_point = candidate
                lo = mid
            else:
                hi = mid
        return np.array(best_point, dtype=np.float32)


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
