"""Tests for labeling methods."""

import pandas as pd
import pytest

from afml.labeling import add_vertical_barrier, apply_pt_sl_on_t1, drop_labels, get_bins, get_events


def _close() -> pd.Series:
    return pd.Series(
        [100.0, 103.0, 99.0, 98.0, 101.0, 105.0],
        index=pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 09:31:00",
                "2024-01-01 09:32:00",
                "2024-01-01 09:33:00",
                "2024-01-01 09:34:00",
                "2024-01-01 09:35:00",
            ],
            name="timestamp",
        ),
    )


def test_add_vertical_barrier_places_first_timestamp_after_horizon() -> None:
    close = _close()
    events = pd.DatetimeIndex(["2024-01-01 09:30:00", "2024-01-01 09:34:00"])

    t1 = add_vertical_barrier(events, close, num_minutes=2)

    assert list(t1) == [pd.Timestamp("2024-01-01 09:32:00"), pd.NaT]
    assert t1.name == "t1"


def test_apply_pt_sl_on_t1_finds_first_horizontal_barrier_touch() -> None:
    close = _close()
    events = pd.DataFrame(
        {
            "t1": pd.to_datetime(["2024-01-01 09:35:00", "2024-01-01 09:35:00"]),
            "trgt": [0.02, 0.02],
            "side": [1.0, 1.0],
        },
        index=pd.DatetimeIndex(["2024-01-01 09:30:00", "2024-01-01 09:32:00"]),
    )

    touches = apply_pt_sl_on_t1(close, events, [1, 1])

    assert touches.loc[pd.Timestamp("2024-01-01 09:30:00"), "pt"] == pd.Timestamp(
        "2024-01-01 09:31:00"
    )
    assert touches.loc[pd.Timestamp("2024-01-01 09:30:00"), "sl"] == pd.Timestamp(
        "2024-01-01 09:33:00"
    )
    assert pd.isna(touches.loc[pd.Timestamp("2024-01-01 09:32:00"), "sl"])
    assert touches.loc[pd.Timestamp("2024-01-01 09:32:00"), "pt"] == pd.Timestamp(
        "2024-01-01 09:34:00"
    )


def test_get_events_returns_first_touch_and_filters_small_targets() -> None:
    close = _close()
    t_events = pd.DatetimeIndex(
        ["2024-01-01 09:30:00", "2024-01-01 09:31:00", "2024-01-01 09:32:00"]
    )
    target = pd.Series(
        [0.02, 0.005, 0.02],
        index=t_events,
        name="trgt",
    )
    t1 = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 09:35:00",
                "2024-01-01 09:35:00",
                "2024-01-01 09:35:00",
            ]
        ),
        index=t_events,
        name="t1",
    )

    events = get_events(close, t_events, 1, target, min_ret=0.01, t1=t1)

    assert list(events.index) == [
        pd.Timestamp("2024-01-01 09:30:00"),
        pd.Timestamp("2024-01-01 09:32:00"),
    ]
    assert events.loc[pd.Timestamp("2024-01-01 09:30:00"), "t1"] == pd.Timestamp(
        "2024-01-01 09:31:00"
    )
    assert events.loc[pd.Timestamp("2024-01-01 09:32:00"), "t1"] == pd.Timestamp(
        "2024-01-01 09:34:00"
    )
    assert "side" not in events.columns


def test_get_events_supports_meta_labeling_side_and_asymmetric_barriers() -> None:
    close = _close()
    t_events = pd.DatetimeIndex(["2024-01-01 09:30:00", "2024-01-01 09:31:00"])
    target = pd.Series([0.02, 0.02], index=t_events, name="trgt")
    side = pd.Series([1.0, -1.0], index=t_events, name="side")
    t1 = pd.Series(pd.to_datetime(["2024-01-01 09:35:00"] * 2), index=t_events, name="t1")

    events = get_events(close, t_events, [1, 2], target, min_ret=0, t1=t1, side=side)

    assert list(events.columns) == ["t1", "trgt", "side"]
    assert events.loc[pd.Timestamp("2024-01-01 09:30:00"), "t1"] == pd.Timestamp(
        "2024-01-01 09:31:00"
    )
    assert events.loc[pd.Timestamp("2024-01-01 09:31:00"), "t1"] == pd.Timestamp(
        "2024-01-01 09:32:00"
    )
    assert list(events["side"]) == [1.0, -1.0]


def test_get_bins_labels_realized_returns_from_events() -> None:
    close = _close()
    events = pd.DataFrame(
        {
            "t1": pd.to_datetime(["2024-01-01 09:31:00", "2024-01-01 09:34:00", pd.NaT]),
            "trgt": [0.02, 0.02, 0.02],
        },
        index=pd.DatetimeIndex(
            ["2024-01-01 09:30:00", "2024-01-01 09:32:00", "2024-01-01 09:33:00"]
        ),
    )

    labels = get_bins(events, close)

    assert list(labels.index) == [
        pd.Timestamp("2024-01-01 09:30:00"),
        pd.Timestamp("2024-01-01 09:32:00"),
    ]
    assert labels.loc[pd.Timestamp("2024-01-01 09:30:00"), "ret"] == pytest.approx(0.03)
    assert labels.loc[pd.Timestamp("2024-01-01 09:30:00"), "bin"] == 1
    assert labels.loc[pd.Timestamp("2024-01-01 09:32:00"), "ret"] == pytest.approx(101 / 99 - 1)
    assert labels.loc[pd.Timestamp("2024-01-01 09:32:00"), "bin"] == 1


def test_get_bins_meta_labels_side_adjusted_pnl() -> None:
    close = _close()
    events = pd.DataFrame(
        {
            "t1": pd.to_datetime(["2024-01-01 09:31:00", "2024-01-01 09:32:00"]),
            "trgt": [0.02, 0.02],
            "side": [1.0, -1.0],
        },
        index=pd.DatetimeIndex(["2024-01-01 09:30:00", "2024-01-01 09:31:00"]),
    )

    labels = get_bins(events, close)

    assert labels.loc[pd.Timestamp("2024-01-01 09:30:00"), "ret"] == pytest.approx(0.03)
    assert labels.loc[pd.Timestamp("2024-01-01 09:30:00"), "bin"] == 1
    assert labels.loc[pd.Timestamp("2024-01-01 09:31:00"), "ret"] == pytest.approx(
        -(99 / 103 - 1)
    )
    assert labels.loc[pd.Timestamp("2024-01-01 09:31:00"), "bin"] == 1

    bad_short = events.copy()
    bad_short.loc[pd.Timestamp("2024-01-01 09:31:00"), "t1"] = pd.Timestamp(
        "2024-01-01 09:35:00"
    )
    labels = get_bins(bad_short, close)

    assert labels.loc[pd.Timestamp("2024-01-01 09:31:00"), "ret"] < 0
    assert labels.loc[pd.Timestamp("2024-01-01 09:31:00"), "bin"] == 0


def test_get_events_then_get_bins_end_to_end() -> None:
    close = _close()
    t_events = pd.DatetimeIndex(["2024-01-01 09:30:00", "2024-01-01 09:31:00"])
    target = pd.Series([0.02, 0.02], index=t_events)
    t1 = pd.Series(pd.to_datetime(["2024-01-01 09:35:00"] * 2), index=t_events)

    events = get_events(close, t_events, 1, target, min_ret=0, t1=t1)
    labels = get_bins(events, close)

    assert events.loc[pd.Timestamp("2024-01-01 09:30:00"), "t1"] == pd.Timestamp(
        "2024-01-01 09:31:00"
    )
    assert labels.loc[pd.Timestamp("2024-01-01 09:30:00"), "bin"] == 1
    assert labels.loc[pd.Timestamp("2024-01-01 09:31:00"), "bin"] == -1


def test_get_events_validates_symmetric_barriers() -> None:
    close = _close()
    events = pd.DataFrame(index=pd.DatetimeIndex(["2024-01-01 09:30:00"]))

    with pytest.raises(ValueError, match="symmetric"):
        get_events(close, events.index, [1, 2], pd.Series([0.01], index=events.index), min_ret=0)

    with pytest.raises(ValueError, match="'t1'"):
        get_bins(events, close)

    side = pd.Series([0.0], index=events.index)
    with pytest.raises(ValueError, match="-1 or 1"):
        get_events(close, events.index, [1, 2], pd.Series([0.01], index=events.index), 0, side=side)


def test_drop_labels_recursively_removes_under_populated_classes() -> None:
    events = pd.DataFrame(
        {
            "bin": [1, 1, 1, 1, 1, -1, -1, -1, 0, 2],
            "ret": range(10),
        }
    )

    filtered = drop_labels(events, min_pct=0.2)

    assert set(filtered["bin"]) == {-1, 1}
    assert len(filtered) == 8


def test_drop_labels_stops_when_only_two_classes_remain() -> None:
    events = pd.DataFrame({"bin": [1, 1, 1, 1, 0]})

    filtered = drop_labels(events, min_pct=0.4)

    assert list(filtered["bin"]) == [1, 1, 1, 1, 0]


def test_drop_labels_validates_inputs() -> None:
    with pytest.raises(ValueError, match="'bin'"):
        drop_labels(pd.DataFrame({"label": [1, 0]}))

    with pytest.raises(ValueError, match="between 0 and 1"):
        drop_labels(pd.DataFrame({"bin": [1, 0]}), min_pct=1.5)
