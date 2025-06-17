import pandas as pd
from importlib import reload

import config
import risk.data as rdata
import risk.var as var
from risk.utils import (
    calculate_daily_returns,
    load_latest_price_data,
    detect_var_breaches,
    summarize_var_breaches,
)
from risk.backtests import kupiec_pof_test


def test_full_pipeline(tmp_path, capsys, monkeypatch):
    # mock yfinance.download
    df = pd.DataFrame(
        {"Close": [100, 101, 102, 103, 104]},
        index=pd.date_range("2024-01-01", periods=5),
    )

    def fake_download(ticker, start=None, end=None):
        return df

    monkeypatch.setattr(rdata.yf, "download", fake_download)
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(var, "PROCESSED_DATA_DIR", str(tmp_path))

    # fetch sample data
    rdata.fetch_and_save_data(
        tickers={"^GSPC": "sp500"},
        start_date="2024-01-01",
        end_date="2024-01-06",
        output_dir=str(tmp_path),
    )

    reload(var)  # pick up patched constant
    var.main()
    out = capsys.readouterr().out
    assert "Historical VaR:" in out
    assert "Parametric VaR:" in out
    assert "Monte Carlo VaR:" in out

    # load CSV and run backtests
    df_loaded = load_latest_price_data(str(tmp_path), "sp500")
    prices = df_loaded.iloc[:, 0]
    rets = calculate_daily_returns(prices)

    df_bt = detect_var_breaches(
        pd.DataFrame({"ret": rets, "var": -abs(rets)}),
        return_col="ret",
        var_col="var",
    )
    summary = summarize_var_breaches(df_bt)
    res = kupiec_pof_test(df_bt["breach"], alpha=config.CONFIDENCE_LEVEL)

    assert summary["count"] == res["x"]
    assert res["n"] == len(df_bt)
