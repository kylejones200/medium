# Did That Policy Actually Work? Causal Inference for Power Sector Analysis

*Using Difference-in-Differences, Synthetic Control, and Propensity
Score Matching to rigorously evaluate environmental policy impacts*

**Kyle Jones**\
13 min read · Oct 6, 2025

------------------------------------------------------------------------

In 2018, several states implemented carbon pricing policies. Five years
later, their emissions are down 15%. Success, right?

Not so fast. Emissions might have declined anyway due to: - Cheaper
natural gas displacing coal - Falling renewable energy costs - Economic
recession reducing electricity demand - Federal efficiency standards for
appliances

How do we know what would have happened *without* the policy? That's the
fundamental problem of causal inference: we can't observe the same state
both with and without treatment.

This article demonstrates three rigorous
methods---Difference-in-Differences, Synthetic Control, and Propensity
Score Matching---to estimate causal effects using 27 years of EPA power
plant data. These techniques power billion-dollar policy decisions and
academic research worldwide.

![Causal inference visualization showing treatment
effects](04_causal_inference_main.png)

## The Fundamental Problem

Imagine California implements carbon pricing in 2018. Emissions drop 20%
by 2023. Did the policy work?

**What we observe:** - California with policy: Emissions down 20%

**What we don't observe but need:** - California without policy: What
would emissions have been?

The **counterfactual** (California without policy) is impossible to
observe---California can't exist in both states simultaneously. Causal
inference constructs this counterfactual from data.

**Naive Comparison Fails:**

You might think: "Compare California to a state without the policy!" But
states differ in many ways: - Different energy mix (CA has more
renewables) - Different economy (CA's GDP growth differs) - Different
regulations (CA has stricter standards pre-policy) - Different weather
(affects electricity demand)

Simply comparing California to, say, Texas would confound the policy
effect with these pre-existing differences. We need methods that account
for this.

## Method 1: Difference-in-Differences (DiD)

DiD is the workhorse of policy evaluation. The intuition is elegant:

1.  Compare California before/after policy (difference #1)
2.  Compare control states before/after same period (difference #2)
3.  Subtract difference #2 from difference #1

The second difference removes trends that would have happened anyway.
What remains is the causal effect.

### The Key Assumption: Parallel Trends

**DiD assumes:** Without treatment, treated and control groups would
have followed parallel trends.

This is testable! Look at pre-treatment trends:

::: {#cb1 .sourceCode}
``` {.sourceCode .python}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load state-level data
states = pd.read_parquet('merged_data/egrid_state_1996-2023.parquet')

# Define treatment (hypothetical carbon pricing states)
treated_states = ['CA', 'NY', 'MA', 'WA', 'OR']
treatment_year = 2018

states['treated'] = states['Plant state abbreviation'].isin(treated_states).astype(int)
states['post'] = (states['data_year'] >= treatment_year).astype(int)

# Calculate carbon intensity
states['carbon_intensity'] = (
    states['State annual CO2 emissions (tons)'] / 
    states['State annual net generation (MWh)']
)

# Check parallel trends visually
pre_period = states[states['data_year'] < treatment_year]

pre_trends = pre_period.groupby(['data_year', 'treated'])['carbon_intensity'].mean().unstack()

plt.figure(figsize=(10, 6))
plt.plot(pre_trends.index, pre_trends[0], 'o-', label='Control States', linewidth=2)
plt.plot(pre_trends.index, pre_trends[1], 's-', label='Treated States', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Carbon Intensity')
plt.title('Pre-Treatment Trends: Are They Parallel?')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('parallel_trends.png', dpi=150)
```
:::

If lines are roughly parallel pre-2018, the parallel trends assumption
holds. If treated states were already declining faster, we can't use
basic DiD.

### Estimating the Treatment Effect

::: {#cb2 .sourceCode}
``` {.sourceCode .python}
import statsmodels.formula.api as smf

# Create interaction term
states['treat_post'] = states['treated'] * states['post']

# DiD regression: Y = β0 + β1*treated + β2*post + β3*(treated*post) + ε
# β3 is the DiD estimate
model = smf.ols(
    'carbon_intensity ~ treated + post + treat_post',
    data=states
).fit(cov_type='cluster', cov_kwds={'groups': states['state_abbr']})

print(model.summary())

did_effect = model.params['treat_post']
did_se = model.bse['treat_post']
did_pval = model.pvalues['treat_post']

print(f"\nTreatment Effect: {did_effect:.6f} tons/MWh")
print(f"Standard Error: {did_se:.6f}")
print(f"P-value: {did_pval:.4f}")

if did_pval < 0.05:
    print(f"\n✓ Policy significantly {'reduced' if did_effect < 0 else 'increased'} emissions")
    print(f"  by {abs(did_effect):.6f} tons/MWh ({abs(did_effect)/states['carbon_intensity'].mean()*100:.1f}%)")
else:
    print("\n✗ No significant policy effect detected")
```
:::

Output might show:

    Treatment Effect: -0.042 tons/MWh
    Standard Error: 0.018
    P-value: 0.020

    ✓ Policy significantly reduced emissions
      by 0.042 tons/MWh (7.8%)

**Interpretation:** Carbon pricing reduced emissions intensity by 0.042
tons/MWh (about 8%), controlling for national trends affecting all
states.

### Event Study: Dynamic Treatment Effects

Was the effect immediate or gradual? Event studies show treatment
effects over time:

::: {#cb4 .sourceCode}
``` {.sourceCode .python}
# Create year dummies relative to treatment
states['years_to_treatment'] = states['data_year'] - treatment_year

# Create interactions (omit year -1 as reference)
for year in range(-5, 6):
    if year != -1:
        states[f'treated_year_{year}'] = (
            states['treated'] * (states['years_to_treatment'] == year)
        ).astype(int)

# Run event study regression
formula = 'carbon_intensity ~ treated + ' + ' + '.join([
    f'treated_year_{y}' for y in range(-5, 6) if y != -1
])

event_model = smf.ols(formula, data=states).fit(cov_type='HC1')

# Extract coefficients
event_time = []
coefficients = []
conf_int = []

for year in range(-5, 6):
    event_time.append(year)
    if year == -1:
        coefficients.append(0)  # Reference period
        conf_int.append((0, 0))
    else:
        coef_name = f'treated_year_{year}'
        coefficients.append(event_model.params.get(coef_name, 0))
        ci = event_model.conf_int().loc[coef_name] if coef_name in event_model.params else (0, 0)
        conf_int.append(ci)

# Plot event study
plt.figure(figsize=(10, 6))
plt.plot(event_time, coefficients, 'o-', linewidth=2, markersize=8)
plt.fill_between(event_time, 
                 [ci[0] for ci in conf_int], 
                 [ci[1] for ci in conf_int], 
                 alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.axvline(-0.5, color='red', linestyle='--', label='Policy Implementation')
plt.xlabel('Years Relative to Policy')
plt.ylabel('Treatment Effect')
plt.title('Event Study: Dynamic Policy Effects')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('event_study.png', dpi=150)
```
:::

**What to look for:** - **Pre-treatment coefficients near zero:**
Confirms parallel trends - **Post-treatment pattern:** Immediate effect?
Gradual? Growing over time? - **Confidence intervals:** Narrow
post-treatment means precise estimates

## Method 2: Synthetic Control

What if you have only ONE treated unit? DiD needs multiple units. Enter
Synthetic Control.

**The idea:** Create a "synthetic California" by weighting control
states to match pre-treatment California as closely as possible. Then
compare real California to synthetic California after treatment.

### Building the Synthetic Control

::: {#cb5 .sourceCode}
``` {.sourceCode .python}
from scipy.optimize import minimize

def synthetic_control(treated_pre, control_pre, control_post):
    """
    Find optimal weights for synthetic control
    
    Returns: weights that minimize distance between treated and synthetic control
    """
    
    def objective(weights):
        synthetic = control_pre @ weights
        return np.sum((treated_pre - synthetic)**2)
    
    n_controls = control_pre.shape[1]
    
    # Constraints: weights sum to 1, all non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_controls)]
    initial = np.ones(n_controls) / n_controls
    
    result = minimize(objective, initial, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

# Prepare data: California as treated unit
ca_data = states[states['Plant state abbreviation'] == 'CA'].sort_values('data_year')
control_data = states[~states['Plant state abbreviation'].isin(treated_states)].sort_values(['Plant state abbreviation', 'data_year'])

# Get pre-treatment outcomes
pre_years = [y for y in range(2012, 2018)]
ca_pre = ca_data[ca_data['data_year'] < 2018]['carbon_intensity'].values

# Build control matrices
control_states_list = control_data['Plant state abbreviation'].unique()
control_pre_matrix = []
control_post_matrix = []

for state in control_states_list:
    state_data = control_data[control_data['Plant state abbreviation'] == state]
    pre = state_data[state_data['data_year'] < 2018]['carbon_intensity'].values
    post = state_data[state_data['data_year'] >= 2018]['carbon_intensity'].values
    
    if len(pre) == len(ca_pre):  # Same length
        control_pre_matrix.append(pre)
        control_post_matrix.append(post)

control_pre_matrix = np.array(control_pre_matrix).T
control_post_matrix = np.array(control_post_matrix).T

# Find optimal weights
weights = synthetic_control(ca_pre, control_pre_matrix, control_post_matrix)

print("Synthetic California composed of:")
for i, weight in enumerate(weights):
    if weight > 0.01:  # Only show states with >1% weight
        state = control_states_list[i]
        print(f"  {state}: {weight*100:.1f}%")
```
:::

Output might show:

    Synthetic California composed of:
      TX: 28.3%
      PA: 22.1%
      FL: 18.7%
      OH: 15.2%
      IL: 10.9%
      GA: 4.8%

Synthetic California is mostly Texas, Pennsylvania, and Florida---states
that, when weighted properly, match California's pre-treatment
trajectory.

### Comparing Real vs Synthetic

::: {#cb7 .sourceCode}
``` {.sourceCode .python}
# Generate synthetic control series
synthetic_ca_pre = control_pre_matrix @ weights
synthetic_ca_post = control_post_matrix @ weights

ca_post = ca_data[ca_data['data_year'] >= 2018]['carbon_intensity'].values

# Calculate treatment effect
gap = ca_post - synthetic_ca_post
avg_effect = gap.mean()

print(f"\nAverage Treatment Effect: {avg_effect:.6f} tons/MWh")

# Visualize
all_years = list(pre_years) + list(range(2018, 2024))

plt.figure(figsize=(12, 6))
plt.plot(pre_years, ca_pre, 'o-', label='California (Actual)', linewidth=2)
plt.plot(range(2018, 2024), ca_post, 'o-', linewidth=2)
plt.plot(pre_years, synthetic_ca_pre, 's--', label='Synthetic California', linewidth=2, color='red')
plt.plot(range(2018, 2024), synthetic_ca_post, 's--', linewidth=2, color='red')
plt.axvline(2017.5, color='black', linestyle='--', alpha=0.5)
plt.fill_between(range(2018, 2024), ca_post, synthetic_ca_post, alpha=0.3, color='green')
plt.xlabel('Year')
plt.ylabel('Carbon Intensity')
plt.title('Synthetic Control: California vs Synthetic California')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('synthetic_control.png', dpi=150)
```
:::

The gap between real and synthetic California after 2018 is the
treatment effect.

### Inference: Placebo Tests

How do we know the effect is real and not chance? Run placebo
tests---apply synthetic control to states that DIDN'T get treatment:

::: {#cb8 .sourceCode}
``` {.sourceCode .python}
# Run synthetic control on each control state (placebos)
placebo_effects = []

for placebo_state in control_states_list[:20]:  # Limit for speed
    # Create synthetic control for this placebo state
    placebo_data = control_data[control_data['Plant state abbreviation'] == placebo_state]
    placebo_pre = placebo_data[placebo_data['data_year'] < 2018]['carbon_intensity'].values
    placebo_post = placebo_data[placebo_data['data_year'] >= 2018]['carbon_intensity'].values
    
    if len(placebo_pre) == len(ca_pre):
        # Build control matrix excluding this placebo
        other_controls_pre = []
        other_controls_post = []
        
        for other_state in control_states_list:
            if other_state != placebo_state:
                other_data = control_data[control_data['Plant state abbreviation'] == other_state]
                pre = other_data[other_data['data_year'] < 2018]['carbon_intensity'].values
                post = other_data[other_data['data_year'] >= 2018]['carbon_intensity'].values
                
                if len(pre) == len(placebo_pre):
                    other_controls_pre.append(pre)
                    other_controls_post.append(post)
        
        if len(other_controls_pre) > 0:
            other_controls_pre = np.array(other_controls_pre).T
            other_controls_post = np.array(other_controls_post).T
            
            placebo_weights = synthetic_control(placebo_pre, other_controls_pre, other_controls_post)
            synthetic_placebo_post = other_controls_post @ placebo_weights
            
            placebo_gap = (placebo_post - synthetic_placebo_post).mean()
            placebo_effects.append(placebo_gap)

# Compare CA effect to placebo distribution
p_value = (np.abs(placebo_effects) >= np.abs(avg_effect)).mean()

print(f"\nPlacebo Test Results:")
print(f"  California effect: {avg_effect:.6f}")
print(f"  Mean placebo effect: {np.mean(placebo_effects):.6f}")
print(f"  P-value: {p_value:.4f}")

if p_value < 0.05:
    print("  ✓ California's effect is statistically significant")
else:
    print("  ✗ Effect not distinguishable from random chance")
```
:::

If California's effect is larger than 95% of placebos, it's
statistically significant at the 5% level.

## Method 3: Propensity Score Matching

DiD and Synthetic Control work at aggregate levels. What about
plant-level interventions?

**Example:** Some plants adopted efficiency upgrades in 2020. Did they
reduce emissions?

Challenge: Plants chose to upgrade. Selection bias---perhaps only
well-run plants with capital upgraded. Comparing upgraders to
non-upgraders confounds treatment effect with plant quality.

**Solution:** Propensity Score Matching creates balanced groups.

### Step 1: Estimate Propensity Scores

::: {#cb9 .sourceCode}
``` {.sourceCode .python}
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Plant-level data (hypothetical treatment)
plants_2020 = plants[plants['data_year'] == 2020].copy()

# Create features predicting treatment
X_features = ['log_generation', 'log_capacity', 'capacity_factor', 'plant_age']

# Simulate treatment based on covariates (in practice, use actual treatment)
np.random.seed(42)
treatment_prob = 1 / (1 + np.exp(-(plants_2020['log_generation'] - 10) / 2))
plants_2020['treated'] = (np.random.random(len(plants_2020)) < treatment_prob).astype(int)

# Outcome: 2021 carbon intensity
plants_2021 = plants[plants['data_year'] == 2021]
outcome_map = plants_2021.set_index('Plant ID')['carbon_intensity']
plants_2020['outcome_2021'] = plants_2020['Plant ID'].map(outcome_map)

# Drop missing
psm_data = plants_2020[X_features + ['treated', 'outcome_2021']].dropna()

# Estimate propensity scores
X = psm_data[X_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(random_state=42)
ps_model.fit(X_scaled, psm_data['treated'])

psm_data['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]

print("Propensity Score Distribution:")
print(psm_data.groupby('treated')['propensity_score'].describe())
```
:::

### Step 2: Check Common Support

Propensity scores should overlap between treated and control. If treated
plants have scores 0.7-0.9 and controls have 0.1-0.3, no overlap---can't
match.

::: {#cb10 .sourceCode}
``` {.sourceCode .python}
plt.figure(figsize=(10, 5))
psm_data[psm_data['treated']==0]['propensity_score'].hist(
    bins=50, alpha=0.5, label='Control', color='blue'
)
psm_data[psm_data['treated']==1]['propensity_score'].hist(
    bins=50, alpha=0.5, label='Treated', color='red'
)
plt.xlabel('Propensity Score')
plt.ylabel('Frequency')
plt.title('Propensity Score Overlap')
plt.legend()
plt.savefig('propensity_overlap.png', dpi=150)
```
:::

Good overlap means matching will work well.

### Step 3: Match and Estimate Effect

::: {#cb11 .sourceCode}
``` {.sourceCode .python}
from sklearn.neighbors import NearestNeighbors

# Nearest neighbor matching
treated = psm_data[psm_data['treated']==1]
control = psm_data[psm_data['treated']==0]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['propensity_score']])

distances, indices = nn.kneighbors(treated[['propensity_score']])

# Get matched pairs
matched_control_idx = control.index[indices.flatten()]
matched_treated_idx = treated.index

# Estimate ATT (Average Treatment Effect on Treated)
treated_outcomes = psm_data.loc[matched_treated_idx, 'outcome_2021']
control_outcomes = psm_data.loc[matched_control_idx, 'outcome_2021']

att = (treated_outcomes.values - control_outcomes.values).mean()
se = (treated_outcomes.values - control_outcomes.values).std() / np.sqrt(len(matched_treated_idx))

print(f"\nAverage Treatment Effect on Treated (ATT): {att:.6f}")
print(f"Standard Error: {se:.6f}")
print(f"95% CI: [{att - 1.96*se:.6f}, {att + 1.96*se:.6f}]")

if abs(att) / se > 1.96:
    print(f"✓ Statistically significant effect at 5% level")
```
:::

### Step 4: Check Balance

Did matching create balanced groups?

::: {#cb12 .sourceCode}
``` {.sourceCode .python}
print("\nCovariate Balance After Matching:")
for var in X_features:
    treated_mean = psm_data.loc[matched_treated_idx, var].mean()
    control_mean = psm_data.loc[matched_control_idx, var].mean()
    pooled_std = psm_data[var].std()
    std_diff = (treated_mean - control_mean) / pooled_std * 100
    
    print(f"  {var}:")
    print(f"    Treated mean: {treated_mean:.4f}")
    print(f"    Control mean: {control_mean:.4f}")
    print(f"    Standardized difference: {std_diff:.2f}%")
    print(f"    {'✓ Good balance' if abs(std_diff) < 10 else '✗ Poor balance'}")
```
:::

Standard differences \< 10% indicate good balance.

## When to Use Which Method?

**Difference-in-Differences:** - ✓ Multiple treated and control units -
✓ Clear before/after period - ✓ Parallel trends assumption holds - ✓
Policy affects groups (states, regions) - Example: State-level carbon
pricing

**Synthetic Control:** - ✓ Single treated unit - ✓ Many control units -
✓ Long pre-treatment period - ✓ Policy affects one entity - Example:
Germany's nuclear phase-out

**Propensity Score Matching:** - ✓ Individual-level treatment - ✓
Selection on observables - ✓ Good covariate overlap - ✓ Cross-sectional
or panel data - Example: Plant efficiency upgrades

## Common Pitfalls

❌ **Assuming correlation = causation:** Just because emissions fell
after a policy doesn't mean the policy caused it.

❌ **Ignoring parallel trends:** If treated group was already trending
differently, DiD fails.

❌ **Poor overlap in PSM:** Can't match if groups are too different.

❌ **P-hacking:** Testing many specifications and reporting the one that
works.

❌ **Confusing statistical and practical significance:** A tiny effect
might be statistically significant but practically meaningless.

✅ **Do:** Pre-register analyses, test assumptions, report robustness
checks, be transparent about limitations.

## So What?

Causal inference transforms policy evaluation from guesswork to science.
Instead of "emissions fell 15% after the policy," we can say "the policy
caused a 7-8% reduction, after accounting for other factors---a \$2.3B
annual benefit."

These methods enable:

**Evidence-based policy:** Know what works before scaling nationwide.
A/B test policies like tech companies A/B test features.

**Accountability:** Did the promised benefits materialize? Rigorous
evaluation holds policymakers accountable.

**Efficient allocation:** Invest in policies with proven effects, not
politically popular but ineffective ones.

**Learning:** Understand why some policies work and others fail. Refine
and improve.

The methods shown here---DiD for group-level policies, Synthetic Control
for single units, PSM for individual interventions---cover most policy
evaluation scenarios. Combined with domain expertise and careful
thinking about identification assumptions, they provide credible answers
to causal questions.

The complete code is in the tutorial. Ready to evaluate your own
policies rigorously? Start with DiD if you have treated and control
groups, synthetic control if you have one treated unit, and PSM for
individual-level interventions. Test your assumptions, check robustness,
and be transparent about limitations.

Correlation isn't causation---but with these tools, you can find out
what is.

------------------------------------------------------------------------

**Causal Inference** · **Policy Analysis** · **Statistics** · **Energy**
· **Python**

------------------------------------------------------------------------

*Found this useful? I'm Kyle Jones---I write about rigorous data
analysis for policy, energy, and climate. Follow for more evidence-based
insights.*
