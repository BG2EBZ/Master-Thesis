from museum_env.env import MuseumEnv


def test_info_status_schema_keys():
    env = MuseumEnv(render_mode=None, enable_event_logs=False)
    try:
        env.reset()
        _, _, _, _, info = env.step(None)
        status = info["status"]
        events = info["events"]

        assert "robot_emotion" in status
        assert "speaker_active" in status
        assert "robot_text_label" in status

        assert "fear_last_response" in status
        assert "fear_last_response_target_idx" in status

        assert "fear_response_move_back" in events
        assert "fear_response_stay" in events
        assert "fear_response_continue_hit" in events

        assert status["robot_text_label"] in {
            "none",
            "explanation",
            "Please_follow_me",
            "I_need_more_space",
        }
    finally:
        env.close()
