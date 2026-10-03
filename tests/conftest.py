import numpy as np
import pandas as pd
import pytest

from autotrader.config import load_config


def make_ohlcv(closes, spread=0.01, start="2022-01-01", volume=1_000_000.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    idx = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": np.full(len(closes), volume)}, index=idx)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
