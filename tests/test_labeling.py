"""Tests for labeling methods."""

import pandas as pd
import pytest

from afml.labeling import add_vertical_barrier, apply_pt_sl_on_t1, triple_barrier_labels


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


def test_triple_barrier_labels_profit_taking_and_stop_loss() -> None:
    close = _close()
    events = pd.DataFrame(
        {
            "t1": pd.to_datetime(["2024-01-01 09:35:00", "2024-01-01 09:34:00"]),
            "trgt": [0.02, 0.02],
            "side": [1.0, 1.0],
        },
        index=pd.DatetimeIndex(["2024-01-01 09:30:00", "2024-01-01 09:31:00"]),
    )

    labels = triple_barrier_labels(close, events, [1, 1, 1])

    assert labels.loc[pd.Timestamp("2024-01-01 09:30:00"), "label"] == 1
    assert labels.loc[pd.Timestamp("2024-01-01 09:30:00"), "ret"] == pytest.approx(0.03)
    assert labels.loc[pd.Timestamp("2024-01-01 09:31:00"), "label"] == -1
    assert labels.loc[pd.Timestamp("2024-01-01 09:31:00"), "ret"] == pytest.approx(99 / 103 - 1)


def test_triple_barrier_labels_uses_vertical_barrier_when_no_horizontal_touch() -> None:
    close = _close()
    events = pd.DataFrame(
        {
            "t1": pd.to_datetime(["2024-01-01 09:32:00"]),
            "trgt": [0.10],
        },
        index=pd.DatetimeIndex(["2024-01-01 09:30:00"]),
    )

    labels = triple_barrier_labels(close, events, [1, 1, 1])

    assert labels["label"].iloc[0] == -1
    assert labels["ret"].iloc[0] == pytest.approx(-0.01)
    assert pd.isna(labels["pt"].iloc[0])
    assert pd.isna(labels["sl"].iloc[0])


def test_triple_barrier_labels_supports_side_adjustment() -> None:
    close = _close()
    events = pd.DataFrame(
        {
            "t1": pd.to_datetime(["2024-01-01 09:35:00"]),
            "trgt": [0.02],
            "side": [-1.0],
        },
        index=pd.DatetimeIndex(["2024-01-01 09:30:00"]),
    )

    labels = triple_barrier_labels(close, events, [1, 1, 1])

    assert labels["sl"].iloc[0] == pd.Timestamp("2024-01-01 09:31:00")
    assert labels["label"].iloc[0] == -1
    assert labels["ret"].iloc[0] == pytest.approx(-0.03)


def test_triple_barrier_labels_can_disable_barriers() -> None:
    close = _close()
    events = pd.DataFrame(
        {
            "t1": pd.to_datetime(["2024-01-01 09:32:00"]),
            "trgt": [0.02],
        },
        index=pd.DatetimeIndex(["2024-01-01 09:30:00"]),
    )

    no_horizontal = triple_barrier_labels(close, events, [0, 0, 1])
    no_vertical = triple_barrier_labels(close, events, [0, 0, 0])

    assert no_horizontal["ret"].iloc[0] == pytest.approx(-0.01)
    assert no_horizontal["label"].iloc[0] == -1
    assert no_vertical["ret"].iloc[0] == pytest.approx(0.05)
    assert no_vertical["label"].iloc[0] == 1


def test_triple_barrier_validates_inputs() -> None:
    close = _close()
    events = pd.DataFrame(index=pd.DatetimeIndex(["2024-01-01 09:30:00"]))

    with pytest.raises(ValueError, match="'trgt'"):
        triple_barrier_labels(close, events, [1, 1, 1])

    with pytest.raises(ValueError, match="pt_sl_t1"):
        triple_barrier_labels(close, pd.DataFrame({"trgt": [0.01]}, index=events.index), [1, 1])
