import threading
import unittest

from test_env import (
    SLEEP_SCALE_MAX,
    SLEEP_SCALE_MIN,
    _update_sleep_scale_from_events,
)


class TestDemoSpeedControls(unittest.TestCase):
    def test_speed_up_doubles_sleep_scale(self):
        speed_up_event = threading.Event()
        speed_down_event = threading.Event()
        speed_up_event.set()

        updated = _update_sleep_scale_from_events(
            1.0,
            speed_up_event=speed_up_event,
            speed_down_event=speed_down_event,
        )

        self.assertAlmostEqual(updated, 2.0)
        self.assertFalse(speed_up_event.is_set())
        self.assertFalse(speed_down_event.is_set())

    def test_speed_down_halves_sleep_scale(self):
        speed_up_event = threading.Event()
        speed_down_event = threading.Event()
        speed_down_event.set()

        updated = _update_sleep_scale_from_events(
            1.0,
            speed_up_event=speed_up_event,
            speed_down_event=speed_down_event,
        )

        self.assertAlmostEqual(updated, 0.5)
        self.assertFalse(speed_up_event.is_set())
        self.assertFalse(speed_down_event.is_set())

    def test_sleep_scale_is_clamped_to_bounds(self):
        speed_up_event = threading.Event()
        speed_down_event = threading.Event()

        speed_up_event.set()
        capped_high = _update_sleep_scale_from_events(
            SLEEP_SCALE_MAX,
            speed_up_event=speed_up_event,
            speed_down_event=speed_down_event,
        )
        self.assertAlmostEqual(capped_high, SLEEP_SCALE_MAX)

        speed_down_event.set()
        capped_low = _update_sleep_scale_from_events(
            SLEEP_SCALE_MIN,
            speed_up_event=speed_up_event,
            speed_down_event=speed_down_event,
        )
        self.assertAlmostEqual(capped_low, SLEEP_SCALE_MIN)

    def test_no_speed_events_keeps_sleep_scale_unchanged(self):
        speed_up_event = threading.Event()
        speed_down_event = threading.Event()

        updated = _update_sleep_scale_from_events(
            1.25,
            speed_up_event=speed_up_event,
            speed_down_event=speed_down_event,
        )

        self.assertAlmostEqual(updated, 1.25)


if __name__ == "__main__":
    unittest.main()
