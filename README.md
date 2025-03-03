# Nvidia Stock Analysis (March 1st, 2023 - March 3rd, 2025)

## Key Takeaways
* The recent dip below both SMAs and MACD bearish crossover hint at a possible trend reversal.
* Nvidia has displayed strong historical outperformance, but volatily remains high.
* Market sentiment has shifted due to AI sector competition and earnings call uncertainty.
* Risk-adjusted returns remain favorable (Sharpe Ratio: 1.83), but investors should be cautious given high drawdowns and expected price volatility.
* Monte Carlo simulations predict significant price fluctuations, reinforcing the need for active risk management.
* The recent earnings announcement has had a significant impact on stock price and trading patterns, underscoring the importance of fundamental analysis alongside technical indicators.



## Trend Analysis with Moving Averages
![Moving Averages](image.png)

Over the past two years, Nvidia stock followed a strong uptrend. However, in recent days, the stock price has fallent below both the 50-day and 200-day Simple Moving Averages (SMA). The proximity of these SMAs also suggest a possible bearish crossover, potentially signaling a trend reversal. If the 50-day SMA crosses bellow the 200-day SMA, futher downside pressure may emerge. However, investors should also watch for any false signals and confirm the trend with additional indicators. The stock's reaction to the late February earnings call has been particularly notable, with significant price movements observed in the days following the announcement.

## Momentum and Overbought/Oversold Conditions
### MACD Analysis
![MACD](image-1.png)

The Moving Average Convergence Divergence (MACD) indicator shows that Nvidia's momentum has recently weakened. The MACD line appears bellow the signal line, which indicates a bearish trend continuation. This might support the bearish crossover observe in the SMAs.

### RSI Analysis
![RSI](image-2.png)

Historically, Nvidia's stock has often been considered ouverbought, frequently exceeding the RSI 70 level before pulling back. However, a notable change has occured since early 2025: despite previous recoveries into overbought territory, the RSI has not demonstrated the same behavior, likely due to investors caution ahead of Nvidia's late February earnings call. This reflects broader marker concerns about the company's competitive position in the AI chip sector, which were adressed in the recent earnings call.

## Correlation Analysis
![Correlation Matrix](image-3.png)

* **Close Price & SMAs:** A strong positve correlation suggests that Nvidia's stock exhibits trend-following behavior, aligning with the moving average patterns observed.

* **Close Price & RSI:** A negative correlation supports a mean-reversion tendency, where price pullbacks occur after overbought conditions.

* **Volume & Daily Returns:** Moderate correlation suggests that higher trading volumes influence short-term price fluctuations but are not the dominant factor in price movement.

## Statistical Analysis
### Daily returns
* **Mean Daily Return:** 0.3% 
* **Daily Volatitity:** 3.23%.

### Hypothesis testing
* **t-statistique:** 2.58
* **p-value:** 0.01

Given the statistically significant p-value, Nvidia's daily returns deviate from a normal distribution, suggesting possible market inefienciencies or external factors driving the stock's performance.

## Risk and Performance Metrics
### Annualized Performance Metruics
* **Annualized Returns:** 93.94%
* **Annualized Volatility:** 51.32%
* **Sharpe Ratio:** 1.8304 (indicating strong risk-adjusted returns)

### Drawdown Analysis
![Drawdowns](image-5.png)

Nvidia expereinced frequent and significant drawdowns, with a **maximum drawdown of -27.05%**. This highlights the stock's high volatility, meaning investors should prepare for sharp price swings.
The period following the earnings announcement showed a significant price movement, contributing to the overall drawdown pattern.

## Market Outperformance
![NVDA vs NASDAQ](image-6.png)

Over the past two years, Nvidia has outperformed the NASDAQ-100, reflecting investors confidence in its growth potential within AI and semiconductor markets.

## Price Prediction Analysis
![ML Model](image-7.png)

* **Mean Squared Error (MSE):** 0.001115 (indicating a relatively accurate model)
* **Model Coefficient:** -0.0565
* **Intercept:** 0.005

The model suggests a potential price correction, but further validation is necessary to confirm predictive power.

## Monte Carlo Simulation for Future Price Estimates
![MonteCarlo](image-8.png)

* **Expected Price *(1 Year Ahead)*:** $300.39
* **Standard Deviation *(Risk)*:** $157.17

The wide standard deviation underscores significant uncertainty in Nvidia's future price movements, reinforcing the importance of risk management strategies for investors.



## Next Steps for Investors
1. Monitor the SMA crossover to confirm a potential downtrend.
2. Assess AI market developments and Nvidia's positioning.
3. Diversify holdings to mitigate Nvidia-specific risks.
4. Analyze the details of the recent earnings report and management's forward guidance to inform investment decisions.
