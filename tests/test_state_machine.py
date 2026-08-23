from dubstudio.jobs.states import can_transition


def test_happy_path():
    assert can_transition("created", "queued")
    assert can_transition("queued", "ingesting")
    assert can_transition("exporting", "completed")
    assert not can_transition("created", "completed")
    assert can_transition("synthesizing", "canceled")


def test_retry():
    assert can_transition("failed", "queued")
