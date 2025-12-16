#!/usr/bin/env python3
"""
Python code extracted from 24_neural_mine_scheduling_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

def generate_pit_block_model(nx=50, ny=50, nz=10, seed=42):
    """
    Generate a synthetic open-pit block model.
    
    Returns:
        DataFrame with columns: block_id, x, y, z, bench, grade_cu_pct, 
                                tonnage, rock_type, mining_cost, processing_cost
    """
    np.random.seed(seed)
    
    blocks = []
    block_id = 0
    
    # Create 3D grid
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                # Only include blocks within pit shell (cone shape)
                r = np.sqrt((x - nx/2)**2 + (y - ny/2)**2)
                max_r = (nx/2) - z * 2  # Narrower at depth
                if r > max_r:
                    continue
                
                blocks.append({
                    'block_id': block_id,
                    'x': x,
                    'y': y,
                    'z': z,
                    'bench': nz - z  # Higher bench number = higher elevation
                })
                block_id += 1
    
    df = pd.DataFrame(blocks)
    n = len(df)
    
    # Generate spatially correlated grades (Gaussian random field)
    grade_field_3d = np.random.randn(nz, ny, nx)
    grade_field_3d = gaussian_filter(grade_field_3d, sigma=3.0)
    
    # Extract grades for valid blocks
    grades = []
    for _, row in df.iterrows():
        grade_base = grade_field_3d[int(row['z']), int(row['y']), int(row['x'])]
        # Transform to realistic copper grades (0.2% - 2.5%)
        grade_cu = 0.8 + 0.5 * grade_base
        grade_cu = np.clip(grade_cu, 0.1, 3.0)
        grades.append(grade_cu)
    
    df['grade_cu_pct'] = grades
    
    # Add high-grade lens (ore body)
    lens_x, lens_y, lens_z = nx/2 + 5, ny/2 - 3, nz/2
    for i, row in df.iterrows():
        dist = np.sqrt((row['x']-lens_x)**2 + (row['y']-lens_y)**2 + (row['z']-lens_z)**2)
        if dist < 8:
            df.at[i, 'grade_cu_pct'] += 1.5 * np.exp(-dist/5)
    
    df['grade_cu_pct'] = np.clip(df['grade_cu_pct'], 0.1, 3.5)
    
    # Tonnage (assume 25m × 25m × 10m blocks, density 2.5 t/m³)
    df['tonnage'] = 25 * 25 * 10 * 2.5
    
    # Rock type (waste vs ore based on cutoff grade)
    df['rock_type'] = df['grade_cu_pct'].apply(lambda g: 'ore' if g > 0.5 else 'waste')
    
    # Costs
    df['mining_cost_per_t'] = 2.5  # $/tonne
    df['processing_cost_per_t'] = df['rock_type'].apply(lambda rt: 15.0 if rt == 'ore' else 0.0)
    
    # Revenue (only for ore)
    cu_price_per_t = 9000  # $/tonne Cu metal
    recovery = 0.85
    df['revenue_per_t'] = df.apply(
        lambda row: (row['grade_cu_pct']/100) * cu_price_per_t * recovery if row['rock_type'] == 'ore' else 0,
        axis=1
    )
    
    # Net value per block
    df['net_value'] = df['tonnage'] * (df['revenue_per_t'] - df['mining_cost_per_t'] - df['processing_cost_per_t'])
    
    return df

# Generate 10 training pits
for i in range(10):
    pit = generate_pit_block_model(nx=40, ny=40, nz=8, seed=100+i)
    pit.to_parquet(f'data/pit_train_{i:03d}.parquet')

# ======================================================================
# Code Block 2
# ======================================================================

def compute_predecessors(df):
    """
    For each block, find all blocks that must be mined before it.
    Simple rule: all blocks in the same (x,y) column with z > this block's z.
    """
    predecessors = {}
    for block_id, row in df.iterrows():
        preds = df[
            (df['x'] == row['x']) & 
            (df['y'] == row['y']) & 
            (df['z'] > row['z'])
        ]['block_id'].tolist()
        predecessors[block_id] = preds
    return predecessors

# ======================================================================
# Code Block 3
# ======================================================================

def compute_legal_mask(df, mined_blocks, remaining_capacity_t, current_blend_sum, current_blend_count, 
                       grade_min=0.8, grade_max=1.2):
    """
    Returns a boolean mask [n_blocks] indicating which blocks can be legally mined this period.
    """
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    
    for i, row in df.iterrows():
        block_id = row['block_id']
        
        # Already mined?
        if block_id in mined_blocks:
            continue
        
        # Predecessors mined?
        preds = predecessors[block_id]
        if not all(p in mined_blocks for p in preds):
            continue
        
        # Capacity check
        if row['tonnage'] > remaining_capacity_t:
            continue
        
        # Blending check (only for ore)
        if row['rock_type'] == 'ore':
            new_blend_sum = current_blend_sum + row['grade_cu_pct'] * row['tonnage']
            new_blend_count = current_blend_count + row['tonnage']
            new_blend_avg = new_blend_sum / max(new_blend_count, 1)
            if new_blend_avg < grade_min or new_blend_avg > grade_max:
                continue
        
        mask[i] = True
    
    return mask

# ======================================================================
# Code Block 4
# ======================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedPointerScheduler(nn.Module):
    """
    Transformer-based mine scheduler with explicit constraint masking.
    """
    def __init__(self, d_in=12, d_model=128, nhead=4, nlayers=3):
        super().__init__()
        
        # Feature encoder
        self.feature_encoder = nn.Linear(d_in, d_model)
        
        # Transformer encoder (learns spatial and geological structure)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # Query token for pointer mechanism
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Projection for attention scoring
        self.score_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, legal_mask):
        """
        Args:
            x: [batch, n_blocks, d_in] - block features
            legal_mask: [batch, n_blocks] - boolean mask of legal moves
        
        Returns:
            probs: [batch, n_blocks] - probability distribution over blocks
        """
        batch_size, n_blocks, _ = x.shape
        
        # Encode features
        h = self.feature_encoder(x)  # [batch, n_blocks, d_model]
        
        # Transformer encoding
        h = self.transformer_encoder(h)  # [batch, n_blocks, d_model]
        
        # Pointer attention
        q = self.query_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        q_proj = self.score_proj(q)  # [batch, 1, d_model]
        
        # Compute attention scores
        scores = torch.matmul(q_proj, h.transpose(1, 2)).squeeze(1)  # [batch, n_blocks]
        
        # Apply legal mask
        scores = scores.masked_fill(~legal_mask, -1e9)
        
        # Softmax to get probabilities
        probs = F.softmax(scores, dim=-1)
        
        return probs

# ======================================================================
# Code Block 5
# ======================================================================

from pyomo.environ import *

def solve_small_pit_milp(df, capacity_per_period=10000, n_periods=50, discount_rate=0.08):
    """
    Solve mine scheduling MILP for a small pit.
    Returns list of (block_id, period) tuples.
    """
    model = ConcreteModel()
    
    blocks = df['block_id'].tolist()
    periods = list(range(n_periods))
    
    # Variables: x[b,t] = 1 if block b mined in period t
    model.x = Var(blocks, periods, domain=Binary)
    
    # Objective: maximize discounted NPV
    def npv_rule(m):
        return sum(
            df.loc[b, 'net_value'] * (1/(1+discount_rate)**t) * m.x[b, t]
            for b in blocks for t in periods
        )
    model.obj = Objective(rule=npv_rule, sense=maximize)
    
    # Constraint: each block mined at most once
    def mine_once_rule(m, b):
        return sum(m.x[b, t] for t in periods) <= 1
    model.mine_once = Constraint(blocks, rule=mine_once_rule)
    
    # Constraint: precedence
    def precedence_rule(m, b, t):
        preds = predecessors[b]
        if not preds:
            return Constraint.Skip
        return m.x[b, t] <= sum(m.x[p, t2] for p in preds for t2 in range(t))
    model.precedence = Constraint(blocks, periods, rule=precedence_rule)
    
    # Constraint: capacity
    def capacity_rule(m, t):
        return sum(df.loc[b, 'tonnage'] * m.x[b, t] for b in blocks) <= capacity_per_period
    model.capacity = Constraint(periods, rule=capacity_rule)
    
    # Solve
    solver = SolverFactory('gurobi')
    result = solver.solve(model)
    
    # Extract schedule
    schedule = []
    for b in blocks:
        for t in periods:
            if model.x[b, t].value > 0.5:
                schedule.append((b, t))
    
    return sorted(schedule, key=lambda x: x[1])

# ======================================================================
# Code Block 6
# ======================================================================

def train_epoch(model, train_pits, optimizer, device='cuda'):
    """
    Train for one epoch over a set of teacher schedules.
    """
    model.train()
    total_loss = 0
    
    for pit_data, teacher_schedule in train_pits:
        df = pit_data
        n_blocks = len(df)
        
        # Prepare static features [n_blocks, d_in]
        features = torch.tensor(df[[
            'grade_cu_pct', 'tonnage', 'bench', 'x', 'y', 'z', 
            'net_value', 'mining_cost_per_t', 'processing_cost_per_t'
        ]].values, dtype=torch.float32).to(device)
        
        # Normalize features
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        
        # Simulate rollout following teacher
        mined_blocks = set()
        remaining_capacity = 10000
        
        for step, (target_block_id, period) in enumerate(teacher_schedule[:100]):  # Limit steps
            # Compute legal mask
            legal_mask_np = compute_legal_mask(df, mined_blocks, remaining_capacity, 0, 0)
            legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool).unsqueeze(0).to(device)
            
            # Add dynamic features (capacity remaining, blocks mined count)
            dynamic_features = torch.tensor([
                [remaining_capacity / 10000, len(mined_blocks) / n_blocks, period / 50]
            ], dtype=torch.float32).to(device)
            dynamic_features = dynamic_features.repeat(n_blocks, 1)
            
            x = torch.cat([features, dynamic_features], dim=-1).unsqueeze(0)  # [1, n_blocks, d_in]
            
            # Forward pass
            probs = model(x, legal_mask)  # [1, n_blocks]
            
            # Teacher target
            target_idx = torch.tensor([target_block_id], dtype=torch.long).to(device)
            
            # Loss: cross-entropy
            loss = F.cross_entropy(probs.log(), target_idx)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update state
            mined_blocks.add(target_block_id)
            remaining_capacity -= df.loc[target_block_id, 'tonnage']
            if remaining_capacity <= 0:
                remaining_capacity = 10000  # Reset for next period
        
    return total_loss / len(train_pits)

# ======================================================================
# Code Block 7
# ======================================================================

def schedule_pit_greedy(model, df, capacity_per_period=10000, n_periods=50, device='cuda'):
    """
    Generate a schedule for a pit using greedy decoding.
    """
    model.eval()
    n_blocks = len(df)
    
    # Prepare static features
    features = torch.tensor(df[[
        'grade_cu_pct', 'tonnage', 'bench', 'x', 'y', 'z', 
        'net_value', 'mining_cost_per_t', 'processing_cost_per_t'
    ]].values, dtype=torch.float32).to(device)
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    schedule = []
    mined_blocks = set()
    
    for period in range(n_periods):
        remaining_capacity = capacity_per_period
        
        while True:
            # Compute legal mask
            legal_mask_np = compute_legal_mask(df, mined_blocks, remaining_capacity, 0, 0)
            if not legal_mask_np.any():
                break  # No legal moves
            
            legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool).unsqueeze(0).to(device)
            
            # Dynamic features
            dynamic_features = torch.tensor([
                [remaining_capacity / capacity_per_period, len(mined_blocks) / n_blocks, period / n_periods]
            ], dtype=torch.float32).to(device)
            dynamic_features = dynamic_features.repeat(n_blocks, 1)
            
            x = torch.cat([features, dynamic_features], dim=-1).unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                probs = model(x, legal_mask)
            
            # Greedy selection
            block_idx = probs.argmax().item()
            block_id = df.iloc[block_idx]['block_id']
            
            # Update state
            schedule.append((block_id, period))
            mined_blocks.add(block_id)
            remaining_capacity -= df.iloc[block_idx]['tonnage']
            
            if remaining_capacity <= 0:
                break
    
    return schedule

# Evaluate NPV
def evaluate_schedule_npv(df, schedule, discount_rate=0.08):
    """Compute NPV of a schedule."""
    npv = 0
    for block_id, period in schedule:
        block_value = df.loc[df['block_id'] == block_id, 'net_value'].values[0]
        discount_factor = 1 / (1 + discount_rate) ** period
        npv += block_value * discount_factor
    return npv

# ======================================================================
# Code Block 8
# ======================================================================

# Use mixed precision for 2× speedup
model = model.half()  # FP16
features = features.half()

# Prune candidate set using fast heuristic
top_k = 500  # Only consider top 500 blocks by value/depth ratio
pruned_indices = df.nlargest(top_k, 'net_value')['block_id'].values

# ======================================================================
# Code Block 9
# ======================================================================

def schedule_pit_beam_search(model, df, beam_width=5, ...):
    """
    Use beam search to explore top-K candidate blocks at each step.
    """
    # Maintain top-K partial schedules
    beams = [{'schedule': [], 'mined': set(), 'score': 0}]
    
    for period in range(n_periods):
        candidates = []
        for beam in beams:
            # Generate all legal next moves
            legal_mask = compute_legal_mask(df, beam['mined'], ...)
            probs = model(x, legal_mask)
            
            # Take top-K moves
            top_k_blocks = probs.topk(beam_width)
            for block_idx, prob in zip(top_k_blocks.indices, top_k_blocks.values):
                new_beam = {
                    'schedule': beam['schedule'] + [(block_idx, period)],
                    'mined': beam['mined'] | {block_idx},
                    'score': beam['score'] + prob.log()
                }
                candidates.append(new_beam)
        
        # Keep top-K beams
        beams = sorted(candidates, key=lambda b: b['score'], reverse=True)[:beam_width]
    
    return beams[0]['schedule']

# ======================================================================
# Code Block 10
# ======================================================================

# Extend block model with 3 stockpile nodes
stockpiles = pd.DataFrame({
    'block_id': ['stockpile_low', 'stockpile_med', 'stockpile_high'],
    'grade_cu_pct': [0.4, 0.8, 1.5],
    'tonnage': [0, 0, 0],  # Accumulates dynamically
    'is_stockpile': True
})
df_extended = pd.concat([df, stockpiles], ignore_index=True)

# ======================================================================
# Code Block 11
# ======================================================================

# Original: maximize NPV
loss_npv = -predicted_npv

# Add: minimize grade variance (smoother mill feed)
loss_grade_var = grade_variance

# Add: minimize number of mining periods (faster payback)
loss_periods = n_periods_used

# Combined
loss = loss_npv + 0.1 * loss_grade_var + 0.05 * loss_periods

# ======================================================================
# Code Block 12
# ======================================================================

# Full implementation: data generation, training, inference

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ============================================================================
# 1. Data Generation
# ============================================================================

def generate_pit_block_model(nx=50, ny=50, nz=10, seed=42):
    """Generate synthetic open-pit block model."""
    np.random.seed(seed)
    
    blocks = []
    block_id = 0
    
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                r = np.sqrt((x - nx/2)**2 + (y - ny/2)**2)
                max_r = (nx/2) - z * 2
                if r > max_r:
                    continue
                
                blocks.append({'block_id': block_id, 'x': x, 'y': y, 'z': z, 
                               'bench': nz - z})
                block_id += 1
    
    df = pd.DataFrame(blocks)
    n = len(df)
    
    # Spatially correlated grades
    grade_field = np.random.randn(nz, ny, nx)
    grade_field = gaussian_filter(grade_field, sigma=3.0)
    
    grades = []
    for _, row in df.iterrows():
        grade_base = grade_field[int(row['z']), int(row['y']), int(row['x'])]
        grade_cu = 0.8 + 0.5 * grade_base
        grades.append(np.clip(grade_cu, 0.1, 3.0))
    
    df['grade_cu_pct'] = grades
    
    # High-grade lens
    lens_x, lens_y, lens_z = nx/2 + 5, ny/2 - 3, nz/2
    for i, row in df.iterrows():
        dist = np.sqrt((row['x']-lens_x)**2 + (row['y']-lens_y)**2 + (row['z']-lens_z)**2)
        if dist < 8:
            df.at[i, 'grade_cu_pct'] += 1.5 * np.exp(-dist/5)
    
    df['grade_cu_pct'] = np.clip(df['grade_cu_pct'], 0.1, 3.5)
    df['tonnage'] = 25 * 25 * 10 * 2.5
    df['rock_type'] = df['grade_cu_pct'].apply(lambda g: 'ore' if g > 0.5 else 'waste')
    df['mining_cost_per_t'] = 2.5
    df['processing_cost_per_t'] = df['rock_type'].apply(lambda rt: 15.0 if rt == 'ore' else 0.0)
    
    cu_price = 9000
    recovery = 0.85
    df['revenue_per_t'] = df.apply(
        lambda row: (row['grade_cu_pct']/100) * cu_price * recovery if row['rock_type'] == 'ore' else 0,
        axis=1
    )
    df['net_value'] = df['tonnage'] * (df['revenue_per_t'] - df['mining_cost_per_t'] - df['processing_cost_per_t'])
    
    return df

# ============================================================================
# 2. Precedence and Masking
# ============================================================================

def compute_predecessors(df):
    """Compute predecessor blocks (same x,y, higher z)."""
    predecessors = {}
    for idx, row in df.iterrows():
        preds = df[
            (df['x'] == row['x']) & 
            (df['y'] == row['y']) & 
            (df['z'] > row['z'])
        ]['block_id'].tolist()
        predecessors[row['block_id']] = preds
    return predecessors

def compute_legal_mask(df, mined_blocks, remaining_capacity):
    """Compute boolean mask of legal blocks."""
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    
    for i, row in df.iterrows():
        bid = row['block_id']
        if bid in mined_blocks:
            continue
        
        preds = predecessors.get(bid, [])
        if not all(p in mined_blocks for p in preds):
            continue
        
        if row['tonnage'] > remaining_capacity:
            continue
        
        mask[i] = True
    
    return mask

# ============================================================================
# 3. Neural Model
# ============================================================================

class MaskedPointerScheduler(nn.Module):
    def __init__(self, d_in=12, d_model=128, nhead=4, nlayers=3):
        super().__init__()
        self.fe = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(encoder_layer, nlayers)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, legal_mask):
        h = self.fe(x)
        h = self.enc(h)
        q = self.query.expand(h.size(0), -1, -1)
        scores = torch.matmul(self.proj(q), h.transpose(1,2)).squeeze(1)
        scores = scores.masked_fill(~legal_mask, -1e9)
        return F.softmax(scores, dim=-1)

# ============================================================================
# 4. Inference
# ============================================================================

def schedule_pit_greedy(model, df, capacity_per_period=10000, n_periods=50, device='cpu'):
    model.eval()
    n_blocks = len(df)
    
    feature_cols = ['grade_cu_pct', 'tonnage', 'bench', 'x', 'y', 'z', 
                    'net_value', 'mining_cost_per_t', 'processing_cost_per_t']
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    schedule = []
    mined_blocks = set()
    
    for period in range(n_periods):
        remaining_capacity = capacity_per_period
        
        while True:
            legal_mask_np = compute_legal_mask(df, mined_blocks, remaining_capacity)
            if not legal_mask_np.any():
                break
            
            legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool).unsqueeze(0).to(device)
            
            dynamic = torch.tensor([[remaining_capacity / capacity_per_period, 
                                     len(mined_blocks) / n_blocks, 
                                     period / n_periods]], dtype=torch.float32).to(device)
            dynamic = dynamic.repeat(n_blocks, 1)
            
            x = torch.cat([features, dynamic], dim=-1).unsqueeze(0)
            
            with torch.no_grad():
                probs = model(x, legal_mask)
            
            block_idx = probs.argmax().item()
            block_id = df.iloc[block_idx]['block_id']
            
            schedule.append((block_id, period))
            mined_blocks.add(block_id)
            remaining_capacity -= df.iloc[block_idx]['tonnage']
            
            if remaining_capacity <= 0:
                break
    
    return schedule

def evaluate_npv(df, schedule, discount_rate=0.08):
    npv = 0
    for block_id, period in schedule:
        val = df[df['block_id'] == block_id]['net_value'].values[0]
        npv += val / (1 + discount_rate) ** period
    return npv

# ============================================================================
# 5. Demo
# ============================================================================

if __name__ == '__main__':
    # Generate test pit
    df = generate_pit_block_model(nx=30, ny=30, nz=6, seed=999)
    predecessors = compute_predecessors(df)
    print(f'Generated pit with {len(df)} blocks')
    
    # Initialize model
    model = MaskedPointerScheduler(d_in=12, d_model=128, nhead=4, nlayers=3)
    
    # In production, load trained weights:
    # model.load_state_dict(torch.load('mine_scheduler.pt'))
    
    # For demo, use untrained model (random policy)
    schedule = schedule_pit_greedy(model, df, capacity_per_period=50000, n_periods=30)
    
    npv = evaluate_npv(df, schedule)
    print(f'Schedule length: {len(schedule)} blocks')
    print(f'NPV: ${npv/1e6:.2f}M')
    
    # Visualize
    df['mining_order'] = -1
    for i, (bid, period) in enumerate(schedule):
        df.loc[df['block_id'] == bid, 'mining_order'] = i
    
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['x'], df['y'], c=df['mining_order'], 
                         cmap='viridis', s=20, edgecolors='black', linewidth=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Mining Order')
    ax.set_xlabel('X (block)')
    ax.set_ylabel('Y (block)')
    ax.set_title('Mine Extraction Sequence (Top View)', fontsize=12, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('mine_schedule_demo.png', dpi=300, bbox_inches='tight')
    print('✓ Visualization saved')

# ======================================================================
# Code Block 13
# ======================================================================

maximize: Σ_{b,t} (revenue[b] - cost[b]) × discount^t × x[b,t]

subject to:
  Σ_b x[b,t] = 1                    for all blocks b (mine once)
  x[b,t] ≤ Σ_{t'<t} x[p,t']         for all predecessors p of b (precedence)
  Σ_b tonnage[b] × x[b,t] ≤ capacity[t]   (capacity)
  grade_min ≤ Σ_b grade[b]×tonnage[b]×x[b,t] / Σ_b tonnage[b]×x[b,t] ≤ grade_max  (blend)

# ======================================================================
# Code Block 14
# ======================================================================

"""
Generate a synthetic open-pit block model.

Returns:
    DataFrame with columns: block_id, x, y, z, bench, grade_cu_pct, 
                            tonnage, rock_type, mining_cost, processing_cost
"""
np.random.seed(seed)

blocks = []
block_id = 0

# ======================================================================
# Code Block 15
# ======================================================================

r = np.sqrt((x - nx/2)**2 + (y - ny/2)**2)
            max_r = (nx/2) - z * 2  # Narrower at depth
            if r > max_r:
                continue
            
            blocks.append({
                'block_id': block_id,
                'x': x,
                'y': y,
                'z': z,
                'bench': nz - z  # Higher bench number = higher elevation
            })
            block_id += 1

df = pd.DataFrame(blocks)
n = len(df)

# ======================================================================
# Code Block 16
# ======================================================================

grade_field_3d = np.random.randn(nz, ny, nx)
grade_field_3d = gaussian_filter(grade_field_3d, sigma=3.0)

# ======================================================================
# Code Block 17
# ======================================================================

grades = []
for _, row in df.iterrows():
    grade_base = grade_field_3d[int(row['z']), int(row['y']), int(row['x'])]

# ======================================================================
# Code Block 18
# ======================================================================

grade_cu = 0.8 + 0.5 * grade_base
    grade_cu = np.clip(grade_cu, 0.1, 3.0)
    grades.append(grade_cu)

df['grade_cu_pct'] = grades

# ======================================================================
# Code Block 19
# ======================================================================

lens_x, lens_y, lens_z = nx/2 + 5, ny/2 - 3, nz/2
for i, row in df.iterrows():
    dist = np.sqrt((row['x']-lens_x)**2 + (row['y']-lens_y)**2 + (row['z']-lens_z)**2)
    if dist < 8:
        df.at[i, 'grade_cu_pct'] += 1.5 * np.exp(-dist/5)

df['grade_cu_pct'] = np.clip(df['grade_cu_pct'], 0.1, 3.5)

# ======================================================================
# Code Block 20
# ======================================================================

df['tonnage'] = 25 * 25 * 10 * 2.5

# ======================================================================
# Code Block 21
# ======================================================================

df['rock_type'] = df['grade_cu_pct'].apply(lambda g: 'ore' if g > 0.5 else 'waste')

# ======================================================================
# Code Block 22
# ======================================================================

df['mining_cost_per_t'] = 2.5  # $/tonne
df['processing_cost_per_t'] = df['rock_type'].apply(lambda rt: 15.0 if rt == 'ore' else 0.0)

# ======================================================================
# Code Block 23
# ======================================================================

cu_price_per_t = 9000  # $/tonne Cu metal
recovery = 0.85
df['revenue_per_t'] = df.apply(
    lambda row: (row['grade_cu_pct']/100) * cu_price_per_t * recovery if row['rock_type'] == 'ore' else 0,
    axis=1
)

# ======================================================================
# Code Block 24
# ======================================================================

df['net_value'] = df['tonnage'] * (df['revenue_per_t'] - df['mining_cost_per_t'] - df['processing_cost_per_t'])

return df

# ======================================================================
# Code Block 25
# ======================================================================

pit = generate_pit_block_model(nx=40, ny=40, nz=8, seed=100+i)
pit.to_parquet(f'data/pit_train_{i:03d}.parquet')

# ======================================================================
# Code Block 26
# ======================================================================

"""
For each block, find all blocks that must be mined before it.
Simple rule: all blocks in the same (x,y) column with z > this block's z.
"""
predecessors = {}
for block_id, row in df.iterrows():
    preds = df[
        (df['x'] == row['x']) & 
        (df['y'] == row['y']) & 
        (df['z'] > row['z'])
    ]['block_id'].tolist()
    predecessors[block_id] = preds
return predecessors

# ======================================================================
# Code Block 27
# ======================================================================

grade_min=0.8, grade_max=1.2):
"""
Returns a boolean mask [n_blocks] indicating which blocks can be legally mined this period.
"""
n = len(df)
mask = np.zeros(n, dtype=bool)

for i, row in df.iterrows():
    block_id = row['block_id']

# ======================================================================
# Code Block 28
# ======================================================================

preds = predecessors[block_id]
    if not all(p in mined_blocks for p in preds):
        continue

# ======================================================================
# Code Block 29
# ======================================================================

if row['rock_type'] == 'ore':
        new_blend_sum = current_blend_sum + row['grade_cu_pct'] * row['tonnage']
        new_blend_count = current_blend_count + row['tonnage']
        new_blend_avg = new_blend_sum / max(new_blend_count, 1)
        if new_blend_avg < grade_min or new_blend_avg > grade_max:
            continue
    
    mask[i] = True

return mask

# ======================================================================
# Code Block 30
# ======================================================================

"""
Transformer-based mine scheduler with explicit constraint masking.
"""
def __init__(self, d_in=12, d_model=128, nhead=4, nlayers=3):
    super().__init__()

# ======================================================================
# Code Block 31
# ======================================================================

self.feature_encoder = nn.Linear(d_in, d_model)

# ======================================================================
# Code Block 32
# ======================================================================

encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=4 * d_model,
        batch_first=True,
        dropout=0.1
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

# ======================================================================
# Code Block 33
# ======================================================================

self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

# ======================================================================
# Code Block 34
# ======================================================================

self.score_proj = nn.Linear(d_model, d_model)

def forward(self, x, legal_mask):
    """
    Args:
        x: [batch, n_blocks, d_in] - block features
        legal_mask: [batch, n_blocks] - boolean mask of legal moves
    
    Returns:
        probs: [batch, n_blocks] - probability distribution over blocks
    """
    batch_size, n_blocks, _ = x.shape

# ======================================================================
# Code Block 35
# ======================================================================

h = self.feature_encoder(x)  # [batch, n_blocks, d_model]

# ======================================================================
# Code Block 36
# ======================================================================

h = self.transformer_encoder(h)  # [batch, n_blocks, d_model]

# ======================================================================
# Code Block 37
# ======================================================================

q = self.query_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
    q_proj = self.score_proj(q)  # [batch, 1, d_model]

# ======================================================================
# Code Block 38
# ======================================================================

scores = torch.matmul(q_proj, h.transpose(1, 2)).squeeze(1)  # [batch, n_blocks]

# ======================================================================
# Code Block 39
# ======================================================================

scores = scores.masked_fill(~legal_mask, -1e9)

# ======================================================================
# Code Block 40
# ======================================================================

probs = F.softmax(scores, dim=-1)
    
    return probs

# ======================================================================
# Code Block 41
# ======================================================================

"""
Solve mine scheduling MILP for a small pit.
Returns list of (block_id, period) tuples.
"""
model = ConcreteModel()

blocks = df['block_id'].tolist()
periods = list(range(n_periods))

# ======================================================================
# Code Block 42
# ======================================================================

model.x = Var(blocks, periods, domain=Binary)

# ======================================================================
# Code Block 43
# ======================================================================

def npv_rule(m):
    return sum(
        df.loc[b, 'net_value'] * (1/(1+discount_rate)**t) * m.x[b, t]
        for b in blocks for t in periods
    )
model.obj = Objective(rule=npv_rule, sense=maximize)

# ======================================================================
# Code Block 44
# ======================================================================

def mine_once_rule(m, b):
    return sum(m.x[b, t] for t in periods) <= 1
model.mine_once = Constraint(blocks, rule=mine_once_rule)

# ======================================================================
# Code Block 45
# ======================================================================

def precedence_rule(m, b, t):
    preds = predecessors[b]
    if not preds:
        return Constraint.Skip
    return m.x[b, t] <= sum(m.x[p, t2] for p in preds for t2 in range(t))
model.precedence = Constraint(blocks, periods, rule=precedence_rule)

# ======================================================================
# Code Block 46
# ======================================================================

def capacity_rule(m, t):
    return sum(df.loc[b, 'tonnage'] * m.x[b, t] for b in blocks) <= capacity_per_period
model.capacity = Constraint(periods, rule=capacity_rule)

# ======================================================================
# Code Block 47
# ======================================================================

solver = SolverFactory('gurobi')
result = solver.solve(model)

# ======================================================================
# Code Block 48
# ======================================================================

schedule = []
for b in blocks:
    for t in periods:
        if model.x[b, t].value > 0.5:
            schedule.append((b, t))

return sorted(schedule, key=lambda x: x[1])

# ======================================================================
# Code Block 49
# ======================================================================

"""
Train for one epoch over a set of teacher schedules.
"""
model.train()
total_loss = 0

for pit_data, teacher_schedule in train_pits:
    df = pit_data
    n_blocks = len(df)

# ======================================================================
# Code Block 50
# ======================================================================

features = torch.tensor(df[[
        'grade_cu_pct', 'tonnage', 'bench', 'x', 'y', 'z', 
        'net_value', 'mining_cost_per_t', 'processing_cost_per_t'
    ]].values, dtype=torch.float32).to(device)

# ======================================================================
# Code Block 51
# ======================================================================

features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

# ======================================================================
# Code Block 52
# ======================================================================

mined_blocks = set()
    remaining_capacity = 10000
    
    for step, (target_block_id, period) in enumerate(teacher_schedule[:100]):  # Limit steps

# ======================================================================
# Code Block 53
# ======================================================================

legal_mask_np = compute_legal_mask(df, mined_blocks, remaining_capacity, 0, 0)
        legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool).unsqueeze(0).to(device)

# ======================================================================
# Code Block 54
# ======================================================================

dynamic_features = torch.tensor([
            [remaining_capacity / 10000, len(mined_blocks) / n_blocks, period / 50]
        ], dtype=torch.float32).to(device)
        dynamic_features = dynamic_features.repeat(n_blocks, 1)
        
        x = torch.cat([features, dynamic_features], dim=-1).unsqueeze(0)  # [1, n_blocks, d_in]

# ======================================================================
# Code Block 55
# ======================================================================

probs = model(x, legal_mask)  # [1, n_blocks]

# ======================================================================
# Code Block 56
# ======================================================================

target_idx = torch.tensor([target_block_id], dtype=torch.long).to(device)

# ======================================================================
# Code Block 57
# ======================================================================

loss = F.cross_entropy(probs.log(), target_idx)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()

# ======================================================================
# Code Block 58
# ======================================================================

mined_blocks.add(target_block_id)
        remaining_capacity -= df.loc[target_block_id, 'tonnage']
        if remaining_capacity <= 0:
            remaining_capacity = 10000  # Reset for next period
    
return total_loss / len(train_pits)

# ======================================================================
# Code Block 59
# ======================================================================

"""
Generate a schedule for a pit using greedy decoding.
"""
model.eval()
n_blocks = len(df)

# ======================================================================
# Code Block 60
# ======================================================================

features = torch.tensor(df[[
    'grade_cu_pct', 'tonnage', 'bench', 'x', 'y', 'z', 
    'net_value', 'mining_cost_per_t', 'processing_cost_per_t'
]].values, dtype=torch.float32).to(device)
features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

schedule = []
mined_blocks = set()

for period in range(n_periods):
    remaining_capacity = capacity_per_period
    
    while True:

# ======================================================================
# Code Block 61
# ======================================================================

legal_mask_np = compute_legal_mask(df, mined_blocks, remaining_capacity, 0, 0)
        if not legal_mask_np.any():
            break  # No legal moves
        
        legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool).unsqueeze(0).to(device)

# ======================================================================
# Code Block 62
# ======================================================================

dynamic_features = torch.tensor([
            [remaining_capacity / capacity_per_period, len(mined_blocks) / n_blocks, period / n_periods]
        ], dtype=torch.float32).to(device)
        dynamic_features = dynamic_features.repeat(n_blocks, 1)
        
        x = torch.cat([features, dynamic_features], dim=-1).unsqueeze(0)

# ======================================================================
# Code Block 63
# ======================================================================

with torch.no_grad():
            probs = model(x, legal_mask)

# ======================================================================
# Code Block 64
# ======================================================================

block_idx = probs.argmax().item()
        block_id = df.iloc[block_idx]['block_id']

# ======================================================================
# Code Block 65
# ======================================================================

schedule.append((block_id, period))
        mined_blocks.add(block_id)
        remaining_capacity -= df.iloc[block_idx]['tonnage']
        
        if remaining_capacity <= 0:
            break

return schedule

# ======================================================================
# Code Block 66
# ======================================================================

"""Compute NPV of a schedule."""
npv = 0
for block_id, period in schedule:
    block_value = df.loc[df['block_id'] == block_id, 'net_value'].values[0]
    discount_factor = 1 / (1 + discount_rate) ** period
    npv += block_value * discount_factor
return npv

# ======================================================================
# Code Block 67
# ======================================================================

beams = [{'schedule': [], 'mined': set(), 'score': 0}]

for period in range(n_periods):
    candidates = []
    for beam in beams:

# ======================================================================
# Code Block 68
# ======================================================================

legal_mask = compute_legal_mask(df, beam['mined'], ...)
        probs = model(x, legal_mask)

# ======================================================================
# Code Block 69
# ======================================================================

top_k_blocks = probs.topk(beam_width)
        for block_idx, prob in zip(top_k_blocks.indices, top_k_blocks.values):
            new_beam = {
                'schedule': beam['schedule'] + [(block_idx, period)],
                'mined': beam['mined'] | {block_idx},
                'score': beam['score'] + prob.log()
            }
            candidates.append(new_beam)

# ======================================================================
# Code Block 70
# ======================================================================

beams = sorted(candidates, key=lambda b: b['score'], reverse=True)[:beam_width]

return beams[0]['schedule']

# ======================================================================
# Code Block 71
# ======================================================================

"""Generate synthetic open-pit block model."""
np.random.seed(seed)

blocks = []
block_id = 0

for z in range(nz):
    for y in range(ny):
        for x in range(nx):
            r = np.sqrt((x - nx/2)**2 + (y - ny/2)**2)
            max_r = (nx/2) - z * 2
            if r > max_r:
                continue
            
            blocks.append({'block_id': block_id, 'x': x, 'y': y, 'z': z, 
                           'bench': nz - z})
            block_id += 1

df = pd.DataFrame(blocks)
n = len(df)

# ======================================================================
# Code Block 72
# ======================================================================

grade_field = np.random.randn(nz, ny, nx)
grade_field = gaussian_filter(grade_field, sigma=3.0)

grades = []
for _, row in df.iterrows():
    grade_base = grade_field[int(row['z']), int(row['y']), int(row['x'])]
    grade_cu = 0.8 + 0.5 * grade_base
    grades.append(np.clip(grade_cu, 0.1, 3.0))

df['grade_cu_pct'] = grades

# ======================================================================
# Code Block 73
# ======================================================================

lens_x, lens_y, lens_z = nx/2 + 5, ny/2 - 3, nz/2
for i, row in df.iterrows():
    dist = np.sqrt((row['x']-lens_x)**2 + (row['y']-lens_y)**2 + (row['z']-lens_z)**2)
    if dist < 8:
        df.at[i, 'grade_cu_pct'] += 1.5 * np.exp(-dist/5)

df['grade_cu_pct'] = np.clip(df['grade_cu_pct'], 0.1, 3.5)
df['tonnage'] = 25 * 25 * 10 * 2.5
df['rock_type'] = df['grade_cu_pct'].apply(lambda g: 'ore' if g > 0.5 else 'waste')
df['mining_cost_per_t'] = 2.5
df['processing_cost_per_t'] = df['rock_type'].apply(lambda rt: 15.0 if rt == 'ore' else 0.0)

cu_price = 9000
recovery = 0.85
df['revenue_per_t'] = df.apply(
    lambda row: (row['grade_cu_pct']/100) * cu_price * recovery if row['rock_type'] == 'ore' else 0,
    axis=1
)
df['net_value'] = df['tonnage'] * (df['revenue_per_t'] - df['mining_cost_per_t'] - df['processing_cost_per_t'])

return df

# ======================================================================
# Code Block 74
# ======================================================================

"""Compute predecessor blocks (same x,y, higher z)."""
predecessors = {}
for idx, row in df.iterrows():
    preds = df[
        (df['x'] == row['x']) & 
        (df['y'] == row['y']) & 
        (df['z'] > row['z'])
    ]['block_id'].tolist()
    predecessors[row['block_id']] = preds
return predecessors

# ======================================================================
# Code Block 75
# ======================================================================

"""Compute boolean mask of legal blocks."""
n = len(df)
mask = np.zeros(n, dtype=bool)

for i, row in df.iterrows():
    bid = row['block_id']
    if bid in mined_blocks:
        continue
    
    preds = predecessors.get(bid, [])
    if not all(p in mined_blocks for p in preds):
        continue
    
    if row['tonnage'] > remaining_capacity:
        continue
    
    mask[i] = True

return mask

# ======================================================================
# Code Block 76
# ======================================================================

def __init__(self, d_in=12, d_model=128, nhead=4, nlayers=3):
    super().__init__()
    self.fe = nn.Linear(d_in, d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, batch_first=True, dropout=0.1)
    self.enc = nn.TransformerEncoder(encoder_layer, nlayers)
    self.query = nn.Parameter(torch.randn(1, 1, d_model))
    self.proj = nn.Linear(d_model, d_model)

def forward(self, x, legal_mask):
    h = self.fe(x)
    h = self.enc(h)
    q = self.query.expand(h.size(0), -1, -1)
    scores = torch.matmul(self.proj(q), h.transpose(1,2)).squeeze(1)
    scores = scores.masked_fill(~legal_mask, -1e9)
    return F.softmax(scores, dim=-1)

# ======================================================================
# Code Block 77
# ======================================================================

model.eval()
n_blocks = len(df)

feature_cols = ['grade_cu_pct', 'tonnage', 'bench', 'x', 'y', 'z', 
                'net_value', 'mining_cost_per_t', 'processing_cost_per_t']
features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

schedule = []
mined_blocks = set()

for period in range(n_periods):
    remaining_capacity = capacity_per_period
    
    while True:
        legal_mask_np = compute_legal_mask(df, mined_blocks, remaining_capacity)
        if not legal_mask_np.any():
            break
        
        legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool).unsqueeze(0).to(device)
        
        dynamic = torch.tensor([[remaining_capacity / capacity_per_period, 
                                 len(mined_blocks) / n_blocks, 
                                 period / n_periods]], dtype=torch.float32).to(device)
        dynamic = dynamic.repeat(n_blocks, 1)
        
        x = torch.cat([features, dynamic], dim=-1).unsqueeze(0)
        
        with torch.no_grad():
            probs = model(x, legal_mask)
        
        block_idx = probs.argmax().item()
        block_id = df.iloc[block_idx]['block_id']
        
        schedule.append((block_id, period))
        mined_blocks.add(block_id)
        remaining_capacity -= df.iloc[block_idx]['tonnage']
        
        if remaining_capacity <= 0:
            break

return schedule

# ======================================================================
# Code Block 78
# ======================================================================

npv = 0
for block_id, period in schedule:
    val = df[df['block_id'] == block_id]['net_value'].values[0]
    npv += val / (1 + discount_rate) ** period
return npv

# ======================================================================
# Code Block 79
# ======================================================================

df = generate_pit_block_model(nx=30, ny=30, nz=6, seed=999)
predecessors = compute_predecessors(df)
print(f'Generated pit with {len(df)} blocks')

# ======================================================================
# Code Block 80
# ======================================================================

model = MaskedPointerScheduler(d_in=12, d_model=128, nhead=4, nlayers=3)

# ======================================================================
# Code Block 81
# ======================================================================

schedule = schedule_pit_greedy(model, df, capacity_per_period=50000, n_periods=30)

npv = evaluate_npv(df, schedule)
print(f'Schedule length: {len(schedule)} blocks')
print(f'NPV: ${npv/1e6:.2f}M')

# ======================================================================
# Code Block 82
# ======================================================================

df['mining_order'] = -1
for i, (bid, period) in enumerate(schedule):
    df.loc[df['block_id'] == bid, 'mining_order'] = i

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df['x'], df['y'], c=df['mining_order'], 
                     cmap='viridis', s=20, edgecolors='black', linewidth=0.3)
cbar = plt.colorbar(scatter, ax=ax, label='Mining Order')
ax.set_xlabel('X (block)')
ax.set_ylabel('Y (block)')
ax.set_title('Mine Extraction Sequence (Top View)', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('mine_schedule_demo.png', dpi=300, bbox_inches='tight')
print('✓ Visualization saved')
