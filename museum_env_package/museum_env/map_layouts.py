from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AxisAlignedRect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def apply_margin(self, margin: float) -> "AxisAlignedRect":
        """Shrink the rectangle by a non-negative margin on each side."""
        m = max(0.0, float(margin))
        return AxisAlignedRect(
            xmin=float(self.xmin) + m,
            xmax=float(self.xmax) - m,
            ymin=float(self.ymin) + m,
            ymax=float(self.ymax) - m,
        )

    def is_valid(self) -> bool:
        return bool(float(self.xmin) <= float(self.xmax) and float(self.ymin) <= float(self.ymax))

    def center(self) -> np.ndarray:
        # Calculate the center point of the rectangle
        return np.array(
            [
                0.5 * (float(self.xmin) + float(self.xmax)),
                0.5 * (float(self.ymin) + float(self.ymax)),
            ],
            dtype=np.float32,
        )

    def area(self) -> float:
        return max(0.0, float(self.xmax) - float(self.xmin)) * max(
            0.0, float(self.ymax) - float(self.ymin)
        )

    def contains_point(self, xy) -> bool:
        x = float(xy[0])
        y = float(xy[1])
        return bool(self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax)

    def sample(self, rng=None) -> np.ndarray:
        rng_choice = np.random if rng is None else rng
        return np.array(
            [
                float(rng_choice.uniform(self.xmin, self.xmax)),
                float(rng_choice.uniform(self.ymin, self.ymax)),
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class MapLayout:
    # Defines the configuration and valid navigation areas for a map.
    name: str
    default_xml_asset: str
    spawn_rects: tuple[AxisAlignedRect, ...]
    robot_waypoints: tuple[tuple[float, float], ...]
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    @lru_cache(maxsize=None)
    def _margin_spawn_rects(self, margin: float) -> tuple[AxisAlignedRect, ...]:
        # Filters and shrinks all spawn areas by the specified margin.
        valid_rects = []
        for rect in self.spawn_rects:
            candidate = rect.apply_margin(margin)
            if candidate.is_valid():
                valid_rects.append(candidate)
        return tuple(valid_rects)

    def get_spawn_rects(self, margin: float) -> tuple[AxisAlignedRect, ...]:
        # Memoized access to margin-adjusted spawn rectangles.
        return self._margin_spawn_rects(float(margin))

    def sample_spawn_point(self, margin: float, rng=None) -> np.ndarray:
        # Picks a random spawn point across all valid rectangles.
        rects = self.get_spawn_rects(margin)
        if not rects:
            if not self.spawn_rects:
                return np.array([0.0, 0.0], dtype=np.float32)
            return self.spawn_rects[0].center()

        rng_choice = np.random if rng is None else rng
        areas = np.array([rect.area() for rect in rects], dtype=np.float64)
        total_area = float(np.sum(areas))
        if total_area <= 0.0:
            return rects[0].center()

        probs = areas / total_area
        rect_idx = int(rng_choice.choice(len(rects), p=probs))
        return rects[rect_idx].sample(rng=rng_choice)


DEFAULT_MUSEUM_LAYOUT = MapLayout(
    name="museum_default",
    default_xml_asset="museum_scene.xml",
    spawn_rects=(
        AxisAlignedRect(0.0, 10.0, 0.0, 10.0),
    ),
    robot_waypoints=(
        (1.0, 5.0),
        (1.0, 4.5),
        (1.0, 2.5),
        (1.5, 2.0),
        (8.5, 2.0),
        (8.5, -10.0),
        (8.5, -12.5),
        (11.0, -12.5),
    ),
    metadata={
        "rooms": ("room_a", "corridor", "room_b"),
        "room_regions": {
            "room_a": AxisAlignedRect(0.0, 10.0, 0.0, 10.0),
            "room_b": AxisAlignedRect(7.0, 12.0, -15.0, -10.0),
        },
        "impatient_corridor_midpoints": {
            "room_a": (8.5, 0.0),
            "room_b": (8.5, -10.0),
        },
        "distractor_exhibit_points": (
            (2.25, 9.785),
            (7.55, 9.785),
            (0.214, 5.00),
            (9.786, 3.35),
            (9.786, 6.65),
            (3.15, 0.214),
            (7.214, -3.25),
            (7.214, -7.35),
            (9.786, -4.65),
            (9.786, -8.35),
            (7.214, -12.95),
            (11.786, -12.50),
            (9.50, -14.786),
        ),
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
