import pandas as pd
from importlib import reload

import risk.var as var
import config


def test_risk_example_runs(tmp_path, capsys, monkeypatch):
    # create simple sp500 CSV
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "close": [1.0, 1.1, 1.2],
        }
    )
    path = tmp_path / "2024-01-01_sp500.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(var, "PROCESSED_DATA_DIR", str(tmp_path))

    reload(var)  # ensure function picks up new constant
    var.main()
    out = capsys.readouterr().out
    assert "Historical VaR:" in out
    assert "Parametric VaR:" in out
    assert "Monte Carlo VaR:" in out
