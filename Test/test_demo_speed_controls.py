import threading
import unittest

from test_env import (
    SLEEP_SCALE_MAX,
    SLEEP_SCALE_MIN,
    _build_hh_distance_mean_1s_extra,
    _build_hr_distance_mean_1s_extra,
    _build_local_crowding_count_1m_extra,
    _combine_periodic_extras,
    _update_sleep_scale_from_events,
    build_arg_parser,
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

    def test_build_hh_distance_mean_1s_extra_formats_all_humans(self):
        info = {
            "metrics": {
                "humans": {
                    "nearest_human_distance_mean_1s": [0.823, 1.04, float("nan")],
                }
            }
        }

        extra = _build_hh_distance_mean_1s_extra(info)

        self.assertEqual(extra, "hh_mean_1s=[h1:0.82, h2:1.04, h3:nan]")

    def test_build_hh_distance_mean_1s_extra_returns_empty_for_missing_metrics(self):
        self.assertEqual(_build_hh_distance_mean_1s_extra({}), "")
        self.assertEqual(_build_hh_distance_mean_1s_extra({"metrics": {}}), "")
        self.assertEqual(_build_hh_distance_mean_1s_extra(None), "")

    def test_build_hr_distance_mean_1s_extra_formats_all_humans(self):
        info = {
            "metrics": {
                "humans": {
                    "human_robot_distance_mean_1s": [1.823, 2.04, float("nan")],
                }
            }
        }

        extra = _build_hr_distance_mean_1s_extra(info)

        self.assertEqual(extra, "hr_mean_1s=[h1:1.82, h2:2.04, h3:nan]")

    def test_build_hr_distance_mean_1s_extra_returns_empty_for_missing_metrics(self):
        self.assertEqual(_build_hr_distance_mean_1s_extra({}), "")
        self.assertEqual(_build_hr_distance_mean_1s_extra({"metrics": {}}), "")
        self.assertEqual(_build_hr_distance_mean_1s_extra(None), "")

    def test_build_local_crowding_count_1m_extra_formats_all_humans(self):
        info = {
            "metrics": {
                "humans": {
                    "local_crowding_count_1m": [4, 7, 2],
                }
            }
        }

        extra = _build_local_crowding_count_1m_extra(info)

        self.assertEqual(extra, "crowd_1m=[h1:4, h2:7, h3:2]")

    def test_build_local_crowding_count_1m_extra_returns_empty_for_missing_metrics(self):
        self.assertEqual(_build_local_crowding_count_1m_extra({}), "")
        self.assertEqual(_build_local_crowding_count_1m_extra({"metrics": {}}), "")
        self.assertEqual(_build_local_crowding_count_1m_extra(None), "")

    def test_combine_periodic_extras_preserves_single_or_multiple_parts(self):
        self.assertEqual(_combine_periodic_extras("", "hh_mean_1s=[h1:1.00]"), "hh_mean_1s=[h1:1.00]")
        self.assertEqual(_combine_periodic_extras("follow_eff/start=[h1:0.1/1.0s]", ""), "follow_eff/start=[h1:0.1/1.0s]")
        self.assertEqual(
            _combine_periodic_extras(
                "follow_eff/start=[h1:0.1/1.0s]",
                "hh_mean_1s=[h1:1.00]",
            ),
            "follow_eff/start=[h1:0.1/1.0s], hh_mean_1s=[h1:1.00]",
        )
        self.assertEqual(
            _combine_periodic_extras(
                "hh_mean_1s=[h1:1.00]",
                "hr_mean_1s=[h1:2.00]",
            ),
            "hh_mean_1s=[h1:1.00], hr_mean_1s=[h1:2.00]",
        )
        self.assertEqual(
            _combine_periodic_extras(
                "",
                "crowd_1m=[h1:4]",
                "",
            ),
            "crowd_1m=[h1:4]",
        )
        self.assertEqual(
            _combine_periodic_extras(
                "hh_mean_1s=[h1:1.00]",
                "hr_mean_1s=[h1:2.00]",
                "crowd_1m=[h1:4]",
            ),
            "hh_mean_1s=[h1:1.00], hr_mean_1s=[h1:2.00], crowd_1m=[h1:4]",
        )

    def test_arg_parser_accepts_hh_distance_mean_flag(self):
        parser = build_arg_parser()

        args = parser.parse_args(["--print-hh-distance-mean-1s"])

        self.assertTrue(args.print_hh_distance_mean_1s)

    def test_arg_parser_accepts_hr_distance_mean_flag(self):
        parser = build_arg_parser()

        args = parser.parse_args(["--print-hr-distance-mean-1s"])

        self.assertTrue(args.print_hr_distance_mean_1s)

    def test_arg_parser_accepts_local_crowding_count_1m_flag(self):
        parser = build_arg_parser()

        args = parser.parse_args(["--print-local-crowding-count-1m"])

        self.assertTrue(args.print_local_crowding_count_1m)


if __name__ == "__main__":
    unittest.main()
