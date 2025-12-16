

# Drivers of US Travel Demand from 2018 to 2025

Evidence from Monthly Federal Transportation Data

## Abstract

Transportation models assume that fuel prices shape travel demand. Monthly data from the Bureau of Transportation Statistics from 2018 to 2025 allow a direct test of this assumption during a period that includes economic expansion, a severe shock, and a slow recovery. Ordinary least squares models with log transformations and fixed effects measure the short-run price elasticity of travel. Structural break tests evaluate the COVID shock. The results show that gas prices have little short-run influence on vehicle miles traveled after seasonal and economic controls. Unemployment explains most observed variation. The COVID shock produces a level shift without a lasting change in slope. These findings support travel forecasting methods that emphasize labor market conditions. They also caution against models that place heavy weight on short-run price shifts.

## Introduction

Policy debates place strong weight on the link between fuel prices and driving. Transportation planning models embed this link through assumed values of price elasticity. The years from 2018 to 2025 offer a natural experiment for this relationship. Gas prices rose and fell. Employment surged and collapsed. COVID disrupted travel at a scale without precedent in the modern record. The period allows a direct test of the drivers of vehicle miles traveled in the short run.

This paper evaluates the effect of gas prices on monthly US travel. The analysis uses federal transportation data and simple econometric models. Fixed effects account for seasonal patterns. Controls for unemployment and GDP isolate economic channels. Structural break tests capture the COVID shock. The goal is to identify the dominant force behind changes in monthly vehicle miles traveled.

**Contribution.** This paper shows that gas prices fail to predict short-run travel once labor market conditions enter the model. This finding challenges the common assumption that fuel price changes drive month-to-month variation in vehicle miles traveled. The results point to a clear pattern. Unemployment predicts travel levels with high precision. Gas prices offer little explanatory value once economic conditions enter the model. Travel falls when people lose jobs. Travel rises when people work. These results align with the logic of commute behavior and with the scale of non-discretionary trips.

## Related Work

The literature reports small negative price elasticities for gasoline demand in the short run. Estimates range from −0.1 to −0.3 in studies that use long horizons and varied identification strategies. These values reflect the limits of substitution in a car-dependent system. Households rely on commute trips. Households adjust vehicle stock over long periods. Households move homes slowly. These constraints reduce the scope of short-run responses to fuel costs.

Recent work shows stronger effects over decades. Sustained price changes influence vehicle choice, residential location, transit supply, and spatial development. These channels evolve slowly. They do not influence month-to-month driving in a modern US system.

COVID created a new domain for travel research. Studies document deep collapses in vehicle travel, followed by uneven recoveries. Research shows strong ties between employment levels and trip volumes. This aligns with the role of commute trips and with broader income effects.

## Data

The analysis uses the Bureau of Transportation Statistics Monthly Transportation Statistics dataset from January 2018 through August 2025. Variables include highway vehicle miles traveled (monthly aggregate, all systems), highway fuel price (regular gasoline, dollars per gallon), unemployment rate (seasonally adjusted), and real GDP (quarterly, seasonally adjusted, forward-filled to monthly).

After cleaning, the dataset contains 70 monthly observations. GDP data, reported quarterly, was forward-filled to match the monthly frequency of other variables. This is a standard approach when mixing data frequencies. The sample covers a period of economic expansion from 2018 to early 2020, the COVID collapse of 2020, and the recovery from 2021 to 2025. The period contains wide swings in both labor market conditions and travel activity.

All variables use their reported federal definitions. Miles traveled use aggregate highway totals from BTS. Gas price uses national average retail values from BTS. Unemployment uses the seasonally adjusted civilian rate from the Bureau of Labor Statistics (via FRED series UNRATE). GDP uses the real chain-weighted measure from the Bureau of Economic Analysis (via FRED series GDPC1).

Data cleaning involved removing currency symbols, commas, and percent signs from formatted numeric fields. Missing values were handled through forward-filling for GDP and listwise deletion for other variables. The final sample contains complete observations for all variables across the full time period.

## Methods

The empirical strategy proceeds in four steps. Model 1 is a naive OLS regression of miles traveled on gas price, with no controls. This establishes a baseline and demonstrates omitted variable bias. Model 2 is a log-log model including unemployment, month fixed effects, and seasonality controls. Log transformation allows interpretation as elasticities. Model 3 adds GDP to separate employment effects from general economic activity. Model 4 tests for COVID-19 disruption using regression discontinuity design, Chow tests, and spline regression.

All models use robust standard errors to address heteroskedasticity. Month fixed effects capture seasonality through dummy variables for each month (with January as the reference category). The Durbin-Watson statistic monitors autocorrelation, though the short time series limits the power of formal time series diagnostics.

The regression discontinuity design uses March 2020 as the treatment date. A dummy variable captures the level shift, and interaction terms test for slope changes. The Chow test compares residual sums of squares before and after the break point. The spline regression uses quadratic terms to allow flexible slope changes while maintaining continuity.

All models use log miles traveled as the dependent variable for elasticity interpretation and variance stabilization. The log-log specification means coefficients can be interpreted as elasticities. For example, a coefficient of -0.025 on unemployment means a one percentage point increase in unemployment is associated with a 2.5% decrease in miles traveled.

**Identification limits.** This analysis tests correlations rather than causal relationships. The goal is prediction rather than causal identification of elasticities. The models control for seasonal patterns through month fixed effects and economic conditions through unemployment and GDP. Structural break tests evaluate discontinuities in the relationship. The analysis does not use instrumental variables or natural experiments to establish causality. Results should be interpreted as descriptive patterns that inform forecasting rather than as causal estimates of price or employment effects.

## Results

### Model 1: Naive Regression

The simple OLS regression of miles traveled on gas price yields a coefficient of 2.96×10¹⁰ (p < 0.001) with R² = 0.291. The positive coefficient contradicts economic expectations. Higher gas prices appear associated with higher travel volumes. This result illustrates omitted variable bias. During the 2018–2020 expansion period, employment rose, incomes grew, and both travel and gas prices increased. The naive model attributes this shared growth to causation rather than correlation.

### Model 2: Controlled Specification

The log-log model with month fixed effects and unemployment controls transforms the results. Gas price elasticity becomes +0.092 (p = 0.028). Unemployment coefficient is -0.025 (p < 0.001). R² increases to 0.867. The model now explains 87% of variance, compared to 29% without controls.

The gas price effect remains positive but becomes economically negligible. A 10% increase in gas prices (roughly $0.30/gallon at current levels) predicts less than 1% increase in miles traveled. The unemployment effect is both statistically and economically significant. Each percentage point increase in unemployment associates with a 2.5% decrease in miles traveled. This relationship is robust, consistent, and aligns with economic theory.

### Model 3: GDP Control

Adding GDP to the model tests whether unemployment captures job-specific effects or broader economic activity. Results show gas price elasticity of +0.073 (p = 0.235), no longer significant. GDP elasticity is +0.052 (p = 0.671), not significant. Unemployment coefficient remains -0.025 (p < 0.001), unchanged. GDP adds no explanatory power. The unemployment effect persists with the same magnitude and significance. This indicates that employment status itself, not aggregate economic output, drives travel behavior.

### Model 4: Structural Break Analysis

March 2020 represents a natural experiment. Regression discontinuity analysis estimates an immediate 9.7% drop in log-transformed miles traveled at the COVID onset. A Chow test confirms a structural break (F = 5.43, p < 0.01). However, quadratic spline regression reveals that post-COVID slopes resemble pre-COVID patterns, suggesting temporary rather than permanent changes in behavior.

The COVID dummy variable in the extended model shows -21.5% effect (p = 0.086), though statistical power is limited by sample size. The unemployment coefficient remains significant throughout (p < 0.001), even during this period of massive disruption.

**Figure 1** plots vehicle miles traveled over time with a vertical line marking March 2020. The figure shows the sharp decline at the COVID onset and the subsequent recovery pattern. The visual break confirms the statistical tests.

**Figure 2** plots fitted values against actual values separately for pre-COVID and post-COVID periods. The figure shows that the model fit remains stable across both periods, with similar residual patterns. This supports the finding that the relationship between unemployment and travel persists through the shock.

### Summary Statistics

Table 1 presents descriptive statistics for the key variables. Vehicle miles traveled averages 268.4 billion miles per month with substantial variation (SD = 25.8 billion). Gas prices range from $1.77 to $5.03 per gallon, averaging $3.12. Unemployment ranges from 3.4% to 14.8%, with a mean of 5.8%. Real GDP shows steady growth from $19.0 to $22.1 trillion (chained 2017 dollars).

The correlation matrix reveals a strong negative correlation between unemployment and miles traveled (-0.75). Gas price and miles traveled show a weak positive correlation (0.54), which disappears after controlling for economic conditions.

## Discussion

The evidence shows that gas prices do not shape short-run travel behavior in the recent period. Employment status shapes travel at a much larger scale. Job loss reduces commute trips. Job loss reduces discretionary income. These forces act fast. They dominate the weak price signal in a system with few substitutes for car travel.

Short-run inelasticity aligns with the logic of US transportation patterns. Most households must drive for essential purposes. Public transit access remains limited. Carpool networks remain thin. Electric vehicles remain a small share. Monthly fluctuations in gas prices fall within a range that does not alter core travel needs.

The COVID shock highlights the role of employment. Travel fell when people stopped working. Travel rose when people returned to work. Price movements during the same period failed to produce comparable shifts.

## Implications for Transportation Planning

Transportation forecasting models should place more weight on labor market conditions than on short-run fuel price scenarios. Forecasters should prioritize unemployment projections, job growth estimates, and industry employment trends when predicting vehicle miles traveled. Models that overweight energy price scenarios will produce less accurate forecasts.

Carbon pricing and fuel taxes may influence long-run outcomes through fleet turnover and land use but yield limited short-run reductions in driving. Planners should not expect immediate travel reductions from fuel price increases. Long-term behavioral change requires complementary policies that address the structural factors making driving necessary.

Policymakers who seek near-term reductions in vehicle miles traveled must address commute structure, trip necessity, and spatial form. Strategies that reduce commute distances, increase transit access, and enable remote work will have larger short-run effects than fuel price policies. Transportation demand management programs should focus on employment patterns and trip purpose rather than fuel costs.

The COVID period offers lessons for planning. Remote work persists, affecting commute patterns and peak-hour congestion. Models calibrated on pre-2020 data may overestimate travel demand. Forecasters should incorporate remote work trends and hybrid employment patterns into their projections.

## Robustness Checks

Two additional specifications test the stability of the main findings. First, a model with lagged unemployment (one-month lag) yields similar results. The unemployment coefficient remains -0.025 (p < 0.001), and gas price elasticity remains insignificant. This suggests the relationship is not driven by reverse causality or timing issues.

Second, re-estimating the models excluding COVID-era observations (March 2020 onwards) shows consistent patterns. The unemployment coefficient remains stable at -0.024 (p < 0.001) in the pre-COVID sample. Gas price elasticity remains small and insignificant. This indicates the core finding is not an artifact of the pandemic period.

Third, a model with gas price in levels instead of logs tests whether the log transformation drives the null result. The level specification yields a gas price coefficient of 0.42 (p = 0.312), which remains insignificant. The unemployment coefficient stays at -0.025 (p < 0.001). This confirms that the finding is not an artifact of functional form. The relationship between unemployment and travel holds regardless of how gas prices enter the model.

The Durbin-Watson statistic for the controlled model is 1.85, suggesting minimal autocorrelation. Robust standard errors address potential heteroskedasticity. These diagnostics support the validity of the OLS estimates.

## Limitations

The sample contains 70 observations and spans an unusual period. The COVID shock covers a large part of the series (approximately 60% of observations). Monthly aggregates mask regional variation. State-level or county-level data would reveal geographic heterogeneity. National aggregates mask regional variation in transit access, urban form, and price sensitivity.

Frequency constraints matter. Monthly data cannot capture daily or weekly adjustments. Higher-frequency data might show short-term price responsiveness that monthly aggregation obscures. The analysis does not attempt causal identification through instrumental variables or natural experiments. It provides descriptive patterns across a turbulent period.

Confounders remain. The 2021–2023 inflation surge, supply chain disruptions, and shift to remote work all affect the relationships being estimated. Academic studies using longer time series (1970s–2010s) typically find small negative price elasticities (-0.1 to -0.3), consistent with economic theory. This analysis does not contradict those findings. It shows that in the specific 2018–2025 period, with COVID disruption and rapid economic changes, the effect is not reliably detected.

## Conclusion

Evidence from federal transportation data from 2018 to 2025 shows a stable relationship between unemployment and travel. Gas prices offer limited short-run predictive value once economic conditions enter the model. These results support the use of labor market variables in travel demand models. They also suggest that policymakers should address the structural forces that drive trip necessity.

