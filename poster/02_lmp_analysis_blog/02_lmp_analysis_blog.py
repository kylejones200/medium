#!/usr/bin/env python3
"""
Python code extracted from 02_lmp_analysis_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
from datetime import datetime, timedelta

def generate_multi_node_lmp(hours=24, seed=42):
    """
    Generate realistic LMP data for multiple market nodes.
    
    Captures hub pricing, constrained zones, renewable zones,
    and urban load centers with realistic price relationships.
    """
    np.random.seed(seed)
    
    node_data = []
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for hour in range(hours):
        timestamp = base_time + timedelta(hours=hour)
        
        # Hub price (baseline reference)
        hub_price = 82.44 + np.random.randn() * 8
        
        # Constrained zone (typically higher due to congestion)
        constraint_premium = 15 + np.random.randn() * 12
        if 16 <= hour <= 21:  # Peak hours see more congestion
            constraint_premium *= 1.5
        constrained_price = hub_price + constraint_premium
        
        # Renewable zone (typically lower due to cheap generation)
        renewable_discount = -8 + np.random.randn() * 6
        renewable_price = hub_price + renewable_discount
        
        # Urban load center (higher due to local demand)
        urban_premium = 8 + np.random.randn() * 10
        if 17 <= hour <= 20:  # Evening peak in cities
            urban_premium *= 1.3
        urban_price = hub_price + urban_premium
        
        node_data.append({
            'timestamp': timestamp,
            'hour': hour,
            'hub_price': hub_price,
            'constrained_zone': constrained_price,
            'renewable_zone': renewable_price,
            'urban_center': urban_price
        })
    
    return node_data

# Generate node pricing
node_prices = generate_multi_node_lmp()
print(f"Hour 19 (Peak):")
print(f"  Hub: ${node_prices[19]['hub_price']:.2f}/MWh")
print(f"  Constrained: ${node_prices[19]['constrained_zone']:.2f}/MWh")
print(f"  Spread: ${node_prices[19]['constrained_zone'] - node_prices[19]['hub_price']:.2f}/MWh")

# ======================================================================
# Code Block 2
# ======================================================================

def calculate_congestion_revenue(node_data):
    """
    Calculate potential congestion revenue from price spreads.
    
    Identifies profitable trading opportunities between nodes
    based on price differentials and typical position sizes.
    """
    opportunities = []
    
    for hour_data in node_data:
        hour = hour_data['hour']
        hub = hour_data['hub_price']
        constrained = hour_data['constrained_zone']
        renewable = hour_data['renewable_zone']
        urban = hour_data['urban_center']
        
        # Arbitrage: Buy hub, sell constrained
        hub_to_constrained_spread = constrained - hub
        
        # Arbitrage: Buy renewable, sell hub
        renewable_to_hub_spread = hub - renewable
        
        # Arbitrage: Buy renewable, sell urban
        renewable_to_urban_spread = urban - renewable
        
        # Calculate potential revenue (assuming 100 MW position)
        position_size_mw = 100
        
        if hub_to_constrained_spread > 10:  # Meaningful spread
            opportunities.append({
                'hour': hour,
                'strategy': 'Hub to Constrained',
                'spread_mwh': hub_to_constrained_spread,
                'revenue_hour': hub_to_constrained_spread * position_size_mw,
                'confidence': 'HIGH' if hub_to_constrained_spread > 20 else 'MEDIUM'
            })
        
        if renewable_to_hub_spread > 5:
            opportunities.append({
                'hour': hour,
                'strategy': 'Renewable to Hub',
                'spread_mwh': renewable_to_hub_spread,
                'revenue_hour': renewable_to_hub_spread * position_size_mw,
                'confidence': 'MEDIUM'
            })
        
        if renewable_to_urban_spread > 15:
            opportunities.append({
                'hour': hour,
                'strategy': 'Renewable to Urban',
                'spread_mwh': renewable_to_urban_spread,
                'revenue_hour': renewable_to_urban_spread * position_size_mw,
                'confidence': 'HIGH' if renewable_to_urban_spread > 25 else 'MEDIUM'
            })
    
    # Calculate daily totals
    total_revenue = sum(opp['revenue_hour'] for opp in opportunities)
    high_confidence_revenue = sum(
        opp['revenue_hour'] for opp in opportunities 
        if opp['confidence'] == 'HIGH'
    )
    
    return {
        'opportunities': opportunities,
        'total_opportunities': len(opportunities),
        'total_daily_revenue': total_revenue,
        'high_confidence_revenue': high_confidence_revenue,
        'avg_spread': total_revenue / len(opportunities) / 100 if opportunities else 0
    }

# Analyze opportunities
results = calculate_congestion_revenue(node_prices)
print(f"\nCongestion Revenue Analysis:")
print(f"  Total Opportunities: {results['total_opportunities']}")
print(f"  Daily Revenue Potential: ${results['total_daily_revenue']:,.0f}")
print(f"  High Confidence Revenue: ${results['high_confidence_revenue']:,.0f}")
print(f"  Average Spread: ${results['avg_spread']:.2f}/MWh")

# ======================================================================
# Code Block 3
# ======================================================================

def monitor_lmp_alerts(current_prices, historical_avg, threshold_std=2.0):
    """
    Monitor LMP for unusual price movements requiring immediate action.
    
    Generates alerts when prices deviate significantly from historical
    patterns, indicating trading opportunities or risk events.
    """
    alerts = []
    
    for node_name, current_price in current_prices.items():
        hist_avg = historical_avg.get(node_name, 85)
        hist_std = 15  # Typical standard deviation
        
        # Calculate z-score
        z_score = (current_price - hist_avg) / hist_std
        
        if abs(z_score) > threshold_std:
            severity = 'CRITICAL' if abs(z_score) > 3 else 'WARNING'
            
            if z_score > 0:
                alert_type = 'SPIKE'
                message = f"Price spike at {node_name}: ${current_price:.2f}/MWh ({z_score:.1f}σ above avg)"
                action = 'SELL' if node_name != 'hub_price' else 'MONITOR'
            else:
                alert_type = 'DROP'
                message = f"Price drop at {node_name}: ${current_price:.2f}/MWh ({abs(z_score):.1f}σ below avg)"
                action = 'BUY' if node_name != 'hub_price' else 'MONITOR'
            
            alerts.append({
                'node': node_name,
                'alert_type': alert_type,
                'severity': severity,
                'current_price': current_price,
                'z_score': z_score,
                'message': message,
                'recommended_action': action
            })
    
    return alerts

# Example: Monitor current market conditions
current_market = {
    'hub_price': 88.50,
    'constrained_zone': 142.30,  # Significant spike
    'renewable_zone': 75.20,
    'urban_center': 94.80
}

historical_average = {
    'hub_price': 85.00,
    'constrained_zone': 95.00,
    'renewable_zone': 78.00,
    'urban_center': 92.00
}

alerts = monitor_lmp_alerts(current_market, historical_average)
print("\nLMP Alert System:")
for alert in alerts:
    print(f"  [{alert['severity']}] {alert['message']}")
    print(f"    Action: {alert['recommended_action']}")

# ======================================================================
# Code Block 4
# ======================================================================

def analyze_ftr_value(node_a, node_b, historical_spreads, ftr_cost):
    """
    Analyze Financial Transmission Right value and profitability.
    
    FTRs pay the difference between nodal prices, providing
    hedge value or speculative profit depending on strategy.
    """
    # Historical spread analysis
    avg_spread = np.mean(historical_spreads)
    std_spread = np.std(historical_spreads)
    max_spread = np.max(historical_spreads)
    min_spread = np.min(historical_spreads)
    
    # Probability of profitability
    profitable_hours = sum(1 for spread in historical_spreads if spread > ftr_cost)
    prob_profitable = profitable_hours / len(historical_spreads)
    
    # Expected value calculation
    expected_revenue = avg_spread
    expected_profit = expected_revenue - ftr_cost
    
    # Risk metrics
    value_at_risk_95 = np.percentile(historical_spreads, 5)
    downside_risk = ftr_cost - value_at_risk_95
    
    return {
        'avg_spread': avg_spread,
        'std_spread': std_spread,
        'max_spread': max_spread,
        'min_spread': min_spread,
        'ftr_cost': ftr_cost,
        'expected_profit': expected_profit,
        'probability_profitable': prob_profitable,
        'var_95': value_at_risk_95,
        'downside_risk': downside_risk,
        'sharpe_ratio': expected_profit / std_spread if std_spread > 0 else 0
    }

# Example: Evaluate FTR purchase decision
historical_hub_to_constrained = [
    15.2, 18.7, 22.3, 12.8, 28.4, 31.2, 19.5, 16.8, 25.1, 20.3,
    14.5, 17.9, 35.6, 21.2, 18.5, 26.7, 19.8, 22.6, 29.3, 17.4,
    15.8, 24.1, 20.7, 18.3, 27.5, 16.2, 21.8, 25.9, 19.1, 23.4
]

ftr_analysis = analyze_ftr_value(
    'hub', 'constrained_zone', 
    historical_hub_to_constrained, 
    ftr_cost=18.50
)

print("\nFTR Analysis (Hub to Constrained Zone):")
print(f"  Average Spread: ${ftr_analysis['avg_spread']:.2f}/MWh")
print(f"  FTR Cost: ${ftr_analysis['ftr_cost']:.2f}/MWh")
print(f"  Expected Profit: ${ftr_analysis['expected_profit']:.2f}/MWh")
print(f"  Probability Profitable: {ftr_analysis['probability_profitable']:.1%}")
print(f"  Sharpe Ratio: {ftr_analysis['sharpe_ratio']:.2f}")
print(f"  VaR (95%): ${ftr_analysis['var_95']:.2f}/MWh")

# ======================================================================
# Code Block 5
# ======================================================================

def decompose_lmp_components(lmp_price, base_energy, transmission_distance_miles):
    """
    Decompose LMP into energy, congestion, and loss components.
    
    Reveals which factors drive price at each node, enabling
    better trading decisions and risk management.
    """
    # Energy component (marginal cost of generation)
    energy_component = base_energy
    
    # Loss component (transmission losses over distance)
    # Typically 1-2% per 100 miles
    loss_rate_per_100_miles = 0.015
    loss_component = base_energy * (transmission_distance_miles / 100) * loss_rate_per_100_miles
    
    # Congestion component (remainder)
    congestion_component = lmp_price - energy_component - loss_component
    
    # Calculate percentages
    energy_pct = (energy_component / lmp_price) * 100
    loss_pct = (loss_component / lmp_price) * 100
    congestion_pct = (congestion_component / lmp_price) * 100
    
    return {
        'total_lmp': lmp_price,
        'energy_component': energy_component,
        'loss_component': loss_component,
        'congestion_component': congestion_component,
        'energy_pct': energy_pct,
        'loss_pct': loss_pct,
        'congestion_pct': congestion_pct,
        'congestion_value_mwh': congestion_component
    }

# Example: Decompose constrained node pricing
decomposition = decompose_lmp_components(
    lmp_price=125.50,
    base_energy=82.00,
    transmission_distance_miles=150
)

print("\nLMP Component Decomposition:")
print(f"  Total LMP: ${decomposition['total_lmp']:.2f}/MWh")
print(f"  Energy: ${decomposition['energy_component']:.2f} ({decomposition['energy_pct']:.1f}%)")
print(f"  Losses: ${decomposition['loss_component']:.2f} ({decomposition['loss_pct']:.1f}%)")
print(f"  Congestion: ${decomposition['congestion_component']:.2f} ({decomposition['congestion_pct']:.1f}%)")
print(f"\n  Congestion represents {decomposition['congestion_pct']:.0f}% of price!")

# ======================================================================
# Code Block 6
# ======================================================================

def build_trading_strategy(node_prices_24h):
    """
    Build optimal trading strategy across multiple nodes.
    
    Considers spreads, volatility, execution costs, and risk limits
    to construct profitable position recommendations.
    """
    trades = []
    
    for hour_data in node_prices_24h:
        hour = hour_data['hour']
        
        # Calculate all possible spreads
        spreads = {
            'hub_to_constrained': hour_data['constrained_zone'] - hour_data['hub_price'],
            'renewable_to_hub': hour_data['hub_price'] - hour_data['renewable_zone'],
            'renewable_to_constrained': hour_data['constrained_zone'] - hour_data['renewable_zone'],
            'hub_to_urban': hour_data['urban_center'] - hour_data['hub_price']
        }
        
        # Transaction costs (bid-ask spread + execution)
        transaction_cost = 2.50  # $/MWh
        
        # Evaluate each spread for trading viability
        for strategy_name, spread in spreads.items():
            net_spread = spread - transaction_cost
            
            if net_spread > 8:  # Minimum profitable threshold
                buy_node, sell_node = strategy_name.split('_to_')
                
                # Position sizing based on spread magnitude
                if net_spread > 20:
                    position_mw = 150
                    confidence = 'HIGH'
                elif net_spread > 12:
                    position_mw = 100
                    confidence = 'MEDIUM'
                else:
                    position_mw = 50
                    confidence = 'LOW'
                
                expected_pnl = net_spread * position_mw
                
                trades.append({
                    'hour': hour,
                    'strategy': strategy_name,
                    'buy_node': buy_node,
                    'sell_node': sell_node,
                    'gross_spread': spread,
                    'net_spread': net_spread,
                    'position_mw': position_mw,
                    'expected_pnl': expected_pnl,
                    'confidence': confidence
                })
    
    # Aggregate results
    total_pnl = sum(t['expected_pnl'] for t in trades)
    high_confidence_trades = [t for t in trades if t['confidence'] == 'HIGH']
    high_confidence_pnl = sum(t['expected_pnl'] for t in high_confidence_trades)
    
    return {
        'trades': trades,
        'total_trades': len(trades),
        'high_confidence_trades': len(high_confidence_trades),
        'total_pnl': total_pnl,
        'high_confidence_pnl': high_confidence_pnl,
        'avg_spread': np.mean([t['net_spread'] for t in trades]) if trades else 0
    }

# Build and evaluate strategy
strategy = build_trading_strategy(node_prices)
print("\nMulti-Node Trading Strategy:")
print(f"  Total Trade Opportunities: {strategy['total_trades']}")
print(f"  High Confidence Trades: {strategy['high_confidence_trades']}")
print(f"  Expected Daily P&L: ${strategy['total_pnl']:,.0f}")
print(f"  High Confidence P&L: ${strategy['high_confidence_pnl']:,.0f}")
print(f"  Average Net Spread: ${strategy['avg_spread']:.2f}/MWh")

# ======================================================================
# Code Block 7
# ======================================================================

"""
Generate realistic LMP data for multiple market nodes.

Captures hub pricing, constrained zones, renewable zones,
and urban load centers with realistic price relationships.
"""
np.random.seed(seed)

node_data = []
base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

for hour in range(hours):
    timestamp = base_time + timedelta(hours=hour)

# ======================================================================
# Code Block 8
# ======================================================================

hub_price = 82.44 + np.random.randn() * 8

# ======================================================================
# Code Block 9
# ======================================================================

constraint_premium = 15 + np.random.randn() * 12
if 16 <= hour <= 21:  # Peak hours see more congestion
        constraint_premium *= 1.5
constrained_price = hub_price + constraint_premium

# ======================================================================
# Code Block 10
# ======================================================================

renewable_discount = -8 + np.random.randn() * 6
renewable_price = hub_price + renewable_discount

# ======================================================================
# Code Block 11
# ======================================================================

urban_premium = 8 + np.random.randn() * 10
if 17 <= hour <= 20:  # Evening peak in cities
        urban_premium *= 1.3
urban_price = hub_price + urban_premium
    
node_data.append({
        'timestamp': timestamp,
        'hour': hour,
        'hub_price': hub_price,
        'constrained_zone': constrained_price,
        'renewable_zone': renewable_price,
        'urban_center': urban_price
    })

return node_data

# ======================================================================
# Code Block 12
# ======================================================================

"""
Calculate potential congestion revenue from price spreads.

Identifies profitable trading opportunities between nodes
based on price differentials and typical position sizes.
"""
opportunities = []

for hour_data in node_data:
    hour = hour_data['hour']
    hub = hour_data['hub_price']
    constrained = hour_data['constrained_zone']
    renewable = hour_data['renewable_zone']
    urban = hour_data['urban_center']

# ======================================================================
# Code Block 13
# ======================================================================

hub_to_constrained_spread = constrained - hub

# ======================================================================
# Code Block 14
# ======================================================================

renewable_to_hub_spread = hub - renewable

# ======================================================================
# Code Block 15
# ======================================================================

renewable_to_urban_spread = urban - renewable

# ======================================================================
# Code Block 16
# ======================================================================

position_size_mw = 100
    
if hub_to_constrained_spread > 10:  # Meaningful spread
        opportunities.append({
            'hour': hour,
            'strategy': 'Hub to Constrained',
            'spread_mwh': hub_to_constrained_spread,
            'revenue_hour': hub_to_constrained_spread * position_size_mw,
            'confidence': 'HIGH' if hub_to_constrained_spread > 20 else 'MEDIUM'
        })
    
if renewable_to_hub_spread > 5:
        opportunities.append({
            'hour': hour,
            'strategy': 'Renewable to Hub',
            'spread_mwh': renewable_to_hub_spread,
            'revenue_hour': renewable_to_hub_spread * position_size_mw,
            'confidence': 'MEDIUM'
        })
    
if renewable_to_urban_spread > 15:
        opportunities.append({
            'hour': hour,
            'strategy': 'Renewable to Urban',
            'spread_mwh': renewable_to_urban_spread,
            'revenue_hour': renewable_to_urban_spread * position_size_mw,
            'confidence': 'HIGH' if renewable_to_urban_spread > 25 else 'MEDIUM'
        })

# ======================================================================
# Code Block 17
# ======================================================================

total_revenue = sum(opp['revenue_hour'] for opp in opportunities)
high_confidence_revenue = sum(
    opp['revenue_hour'] for opp in opportunities 
    if opp['confidence'] == 'HIGH'
)

return {
    'opportunities': opportunities,
    'total_opportunities': len(opportunities),
    'total_daily_revenue': total_revenue,
    'high_confidence_revenue': high_confidence_revenue,
    'avg_spread': total_revenue / len(opportunities) / 100 if opportunities else 0
}

# ======================================================================
# Code Block 18
# ======================================================================

"""
Monitor LMP for unusual price movements requiring immediate action.

Generates alerts when prices deviate significantly from historical
patterns, indicating trading opportunities or risk events.
"""
alerts = []

for node_name, current_price in current_prices.items():
    hist_avg = historical_avg.get(node_name, 85)
    hist_std = 15  # Typical standard deviation

# ======================================================================
# Code Block 19
# ======================================================================

z_score = (current_price - hist_avg) / hist_std
    
if abs(z_score) > threshold_std:
        severity = 'CRITICAL' if abs(z_score) > 3 else 'WARNING'
        
        if z_score > 0:
            alert_type = 'SPIKE'
            message = f"Price spike at {node_name}: ${current_price:.2f}/MWh ({z_score:.1f}σ above avg)"
            action = 'SELL' if node_name != 'hub_price' else 'MONITOR'
        else:
            alert_type = 'DROP'
            message = f"Price drop at {node_name}: ${current_price:.2f}/MWh ({abs(z_score):.1f}σ below avg)"
            action = 'BUY' if node_name != 'hub_price' else 'MONITOR'
        
        alerts.append({
            'node': node_name,
            'alert_type': alert_type,
            'severity': severity,
            'current_price': current_price,
            'z_score': z_score,
            'message': message,
            'recommended_action': action
        })

return alerts

# ======================================================================
# Code Block 20
# ======================================================================

print(f"  [{alert['severity']}] {alert['message']}")
print(f"    Action: {alert['recommended_action']}")

# ======================================================================
# Code Block 21
# ======================================================================

avg_spread = np.mean(historical_spreads)
std_spread = np.std(historical_spreads)
max_spread = np.max(historical_spreads)
min_spread = np.min(historical_spreads)

# ======================================================================
# Code Block 22
# ======================================================================

profitable_hours = sum(1 for spread in historical_spreads if spread > ftr_cost)
prob_profitable = profitable_hours / len(historical_spreads)

# ======================================================================
# Code Block 23
# ======================================================================

expected_revenue = avg_spread
expected_profit = expected_revenue - ftr_cost

# ======================================================================
# Code Block 24
# ======================================================================

value_at_risk_95 = np.percentile(historical_spreads, 5)
downside_risk = ftr_cost - value_at_risk_95

return {
    'avg_spread': avg_spread,
    'std_spread': std_spread,
    'max_spread': max_spread,
    'min_spread': min_spread,
    'ftr_cost': ftr_cost,
    'expected_profit': expected_profit,
    'probability_profitable': prob_profitable,
    'var_95': value_at_risk_95,
    'downside_risk': downside_risk,
    'sharpe_ratio': expected_profit / std_spread if std_spread > 0 else 0
}

# ======================================================================
# Code Block 25
# ======================================================================

'hub', 'constrained_zone', 
historical_hub_to_constrained, 
ftr_cost=18.50

# ======================================================================
# Code Block 26
# ======================================================================

energy_component = base_energy

# ======================================================================
# Code Block 27
# ======================================================================

loss_rate_per_100_miles = 0.015
loss_component = base_energy * (transmission_distance_miles / 100) * loss_rate_per_100_miles

# ======================================================================
# Code Block 28
# ======================================================================

congestion_component = lmp_price - energy_component - loss_component

# ======================================================================
# Code Block 29
# ======================================================================

energy_pct = (energy_component / lmp_price) * 100
loss_pct = (loss_component / lmp_price) * 100
congestion_pct = (congestion_component / lmp_price) * 100

return {
    'total_lmp': lmp_price,
    'energy_component': energy_component,
    'loss_component': loss_component,
    'congestion_component': congestion_component,
    'energy_pct': energy_pct,
    'loss_pct': loss_pct,
    'congestion_pct': congestion_pct,
    'congestion_value_mwh': congestion_component
}

# ======================================================================
# Code Block 30
# ======================================================================

lmp_price=125.50,
base_energy=82.00,
transmission_distance_miles=150

# ======================================================================
# Code Block 31
# ======================================================================

"""
Build optimal trading strategy across multiple nodes.

Considers spreads, volatility, execution costs, and risk limits
to construct profitable position recommendations.
"""
trades = []

for hour_data in node_prices_24h:
    hour = hour_data['hour']

# ======================================================================
# Code Block 32
# ======================================================================

spreads = {
        'hub_to_constrained': hour_data['constrained_zone'] - hour_data['hub_price'],
        'renewable_to_hub': hour_data['hub_price'] - hour_data['renewable_zone'],
        'renewable_to_constrained': hour_data['constrained_zone'] - hour_data['renewable_zone'],
        'hub_to_urban': hour_data['urban_center'] - hour_data['hub_price']
    }

# ======================================================================
# Code Block 33
# ======================================================================

transaction_cost = 2.50  # $/MWh

# ======================================================================
# Code Block 34
# ======================================================================

for strategy_name, spread in spreads.items():
        net_spread = spread - transaction_cost
        
        if net_spread > 8:  # Minimum profitable threshold
            buy_node, sell_node = strategy_name.split('_to_')

# ======================================================================
# Code Block 35
# ======================================================================

if net_spread > 20:
                position_mw = 150
                confidence = 'HIGH'
elif net_spread > 12:
                position_mw = 100
                confidence = 'MEDIUM'
else:
                position_mw = 50
                confidence = 'LOW'
            
expected_pnl = net_spread * position_mw
            
trades.append({
                'hour': hour,
                'strategy': strategy_name,
                'buy_node': buy_node,
                'sell_node': sell_node,
                'gross_spread': spread,
                'net_spread': net_spread,
                'position_mw': position_mw,
                'expected_pnl': expected_pnl,
                'confidence': confidence
            })

# ======================================================================
# Code Block 36
# ======================================================================

total_pnl = sum(t['expected_pnl'] for t in trades)
high_confidence_trades = [t for t in trades if t['confidence'] == 'HIGH']
high_confidence_pnl = sum(t['expected_pnl'] for t in high_confidence_trades)

return {
    'trades': trades,
    'total_trades': len(trades),
    'high_confidence_trades': len(high_confidence_trades),
    'total_pnl': total_pnl,
    'high_confidence_pnl': high_confidence_pnl,
    'avg_spread': np.mean([t['net_spread'] for t in trades]) if trades else 0
}
