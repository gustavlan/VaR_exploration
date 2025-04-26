# Quantitative VaR Backtesting Framework

## 1. Overview

This repository implements a production-grade pipeline for computing and backtesting **10-day Value-at-Risk (VaR)** for major equity indices (S&P 500, NASDAQ 100). Core features include:

1. **Historical, Parametric & EWMA VaR**  
2. **Kupiec Proportion-of-Failures (POF) & Christoffersen independence tests**  
3. **Out-of-sample rolling-window backtest**  
4. **Expected Shortfall (ES / CVaR) comparison**  
5. **Automated CI / linting / unit tests**

Parameters (confidence level, horizon, EWMA λ) live in **`config.py`** for rapid scenario analysis.

---

## 2. Theoretical Background

### 2.1 Value-at-Risk (VaR)

For a return series \(r_t\), the one-period VaR at confidence level \(\alpha\) is
\[
  \mathrm{VaR}_\alpha = -\inf\{x : F_{r}(x) \ge \alpha\},
\]
where \(F_r\) is the cumulative distribution function of returns.  

- **Historical VaR** uses empirical quantiles:  
  \[
    \widehat{\mathrm{VaR}}_\alpha^{\mathrm{hist}} = -Q_{\alpha}(r_t).
  \]
- **Parametric (Gaussian) VaR** assumes \(r_t \sim \mathcal{N}(\mu,\sigma^2)\):  
  \[
    \widehat{\mathrm{VaR}}_\alpha^{\mathrm{param}} 
    = -\bigl(\mu + z_\alpha\,\sigma\bigr),\quad z_\alpha = \Phi^{-1}(\alpha).
  \]
- **EWMA Volatility** (RiskMetrics):  
  \[
    \sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2,\quad \lambda\in(0,1),
  \]
  and  
  \[
    \widehat{\mathrm{VaR}}_\alpha^{\mathrm{EWMA}} 
    = -z_\alpha\,\sqrt{h\,\sigma_t^2},
  \]
  for a horizon of \(h\) days.

### 2.2 Expected Shortfall (ES / CVaR)

While VaR captures a quantile, **ES** measures the average loss beyond that threshold:
\[
  \mathrm{ES}_\alpha = \mathbb{E}\bigl[-r_t \mid r_t \le -\mathrm{VaR}_\alpha\bigr].
\]
ES is a coherent risk measure (subadditive) and often preferred by regulators.

---

## 3. Backtesting Framework

### 3.1 Kupiec POF Test (Unconditional Coverage)

Kupiec (1995) evaluates whether the observed breach rate \(\hat p = x/n\) matches the nominal \(p = 1-\alpha\). The likelihood-ratio statistic is
\[
  \mathrm{LR}_{\mathrm{POF}}
  = -2 \ln
    \frac{(1-p)^{n-x}\,p^{x}}
         {(1-\hat p)^{n-x}\,\hat p^{x}}
  \;\sim\;\chi^2_1.
\]

### 3.2 Christoffersen Independence Test

Christoffersen (1998) tests the null hypothesis that breaches are independent by forming the 2×2 transition counts \(\{N_{00},N_{01},N_{10},N_{11}\}\) and comparing the likelihood of a one-state vs. two-state Markov model:
\[
  \mathrm{LR}_{\mathrm{Ind}}
  = -2\bigl[\ell_{\mathrm{ind}} - \ell_{\mathrm{markov}}\bigr]
  \;\sim\;\chi^2_1.
\]

### 3.3 Out-of-Sample Rolling-Window

Rather than fitting VaR once on the full sample, we:

1. Choose window length \(W\) (e.g. 500 trading days).  
2. For each date \(t \ge W\):  
   - **Train** on \(\{r_{t-W},\dots,r_{t-1}\}\), estimate \(\sigma_{t-1}\) and VaR\(_{t-1}\).  
   - **Test** breach on \(r_t\).  
3. Aggregate out-of-sample breaches and perform Kupiec & Christoffersen tests on that series.

---


## 4. Project Structure

```
.
├── src/
│   └── risk/
│       ├── __init__.py
│       ├── data.py           # Data fetching
│       ├── utils.py          # Returns/vol/VAR helpers
│       ├── var.py            # Historical/parametric/MC VaR
│       └── backtests.py      # POF, Christoffersen, ES
├── notebooks/                # Jupyter analysis
│   └── VaR_exploration.ipynb
├── tests/
│   ├── test_utils.py
│   └── test_var.py
├── requirements.txt
├── pytest.ini
```


## 5. References

1. Kupiec, P. (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models*.  
2. Christoffersen, P. F. (1998). *Evaluating Interval Forecasts*.  
3. J.P. Morgan (1996). *RiskMetrics – Technical Document*.  
4. Acerbi, C. & Tasche, D. (2002). *On the Coherence of Expected Shortfall*.