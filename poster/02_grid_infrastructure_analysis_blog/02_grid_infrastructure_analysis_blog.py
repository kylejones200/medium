#!/usr/bin/env python3
"""
Python code extracted from 02_grid_infrastructure_analysis_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class TransmissionLinesService:
    """Service for handling transmission line data and grid visualization."""
    
    def __init__(self, data_path: str = "HIFLD_Transmission_Lines.parquet"):
        """Initialize transmission lines service.
        
        Args:
            data_path: Path to the transmission lines data file.
        """
        self.data_path = data_path
        self.lines_data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load transmission lines data from Parquet or CSV."""
        try:
            if os.path.exists(self.data_path):
                logger.info(f"Loading transmission lines data from {self.data_path}")
                
                # Load based on file extension
                if self.data_path.endswith('.parquet'):
                    self.lines_data = pd.read_parquet(self.data_path)
                else:
                    self.lines_data = pd.read_csv(self.data_path)
                
                logger.info(f"Loaded {len(self.lines_data):,} transmission lines")
                
                # Standardize column names
                if 'START_LAT' in self.lines_data.columns:
                    self.lines_data['latitude'] = self.lines_data['START_LAT']
                    self.lines_data['longitude'] = self.lines_data['START_LON']
                
            else:
                logger.warning(f"Transmission lines file not found: {self.data_path}")
                self.lines_data = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load transmission lines data: {e}")
            self.lines_data = pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the transmission lines data."""
        if self.lines_data is None or self.lines_data.empty:
            return {}
        
        try:
            # Calculate comprehensive statistics
            stats = {
                "total_lines": len(self.lines_data),
                "in_service": len(self.lines_data[self.lines_data['STATUS'] == 'IN SERVICE']),
                "voltage_classes": len(self.lines_data['VOLT_CLASS'].dropna().unique()),
                "unique_owners": len(self.lines_data['OWNER'].dropna().unique()),
                "total_length_miles": self.lines_data.get('SHAPE_Length', 
                                                         self.lines_data.get('SHAPE__Length', 
                                                         pd.Series([0]))).sum() / 5280,
                "avg_voltage": self.lines_data['VOLTAGE'].replace(-999999, None).mean(),
                "overhead_lines": len(self.lines_data[self.lines_data['TYPE'].str.contains('OVERHEAD', na=False)]),
                "underground_lines": len(self.lines_data[self.lines_data['TYPE'].str.contains('UNDERGROUND', na=False)])
            }
            
            # Add voltage class breakdown
            voltage_distribution = self.get_voltage_class_distribution()
            stats['voltage_distribution'] = voltage_distribution
            
            # Add geographic coverage
            if 'STATE' in self.lines_data.columns:
                stats['states_covered'] = self.lines_data['STATE'].nunique()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return {}
    
    def get_voltage_class_distribution(self) -> List[Dict[str, Any]]:
        """Get voltage class distribution with counts."""
        if self.lines_data is None or self.lines_data.empty:
            return []
        
        try:
            # Filter out invalid voltage classes
            valid_data = self.lines_data[
                (self.lines_data['VOLT_CLASS'].notna()) & 
                (self.lines_data['VOLT_CLASS'] != '-999999')
            ]
            
            voltage_counts = valid_data['VOLT_CLASS'].value_counts().head(10)
            
            return [
                {"voltage_class": str(vc), "count": int(count), "percentage": round(count/len(valid_data)*100, 2)}
                for vc, count in voltage_counts.items()
            ]
            
        except Exception as e:
            logger.error(f"Failed to get voltage distribution: {e}")
            return []
    
    def get_lines_by_voltage_class(self, voltage_class: str = "345-765", limit: int = 1000) -> List[Dict[str, Any]]:
        """Get transmission lines filtered by voltage class.
        
        Args:
            voltage_class: Voltage class to filter by (e.g., '345-765', '220-287')
            limit: Maximum number of lines to return.
            
        Returns:
            List of transmission line records.
        """
        if self.lines_data is None or self.lines_data.empty:
            return []
        
        try:
            filtered_lines = self.lines_data[
                self.lines_data['VOLT_CLASS'] == voltage_class
            ].head(limit)
            
            return filtered_lines.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to filter transmission lines: {e}")
            return []
    
    def get_lines_by_owner(self, owner: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get transmission lines filtered by owner/utility.
        
        Args:
            owner: Owner/utility name to filter by.
            limit: Maximum number of lines to return.
            
        Returns:
            List of transmission line records.
        """
        if self.lines_data is None or self.lines_data.empty:
            return []
        
        try:
            filtered_lines = self.lines_data[
                self.lines_data['OWNER'].str.contains(owner, case=False, na=False)
            ].head(limit)
            
            return filtered_lines.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to filter by owner: {e}")
            return []
    
    def get_major_utilities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get major utilities ranked by transmission line count.
        
        Args:
            limit: Maximum number of utilities to return.
            
        Returns:
            List of utilities with line counts.
        """
        if self.lines_data is None or self.lines_data.empty:
            return []
        
        try:
            # Get top utilities by line count
            utility_counts = self.lines_data['OWNER'].value_counts().head(limit)
            
            utilities = []
            for owner, count in utility_counts.items():
                if owner not in ['NOT AVAILABLE', 'UNKNOWN', '-999999']:
                    # Calculate additional statistics for this utility
                    utility_lines = self.lines_data[self.lines_data['OWNER'] == owner]
                    
                    utilities.append({
                        'name': owner,
                        'line_count': int(count),
                        'total_miles': round(utility_lines.get('SHAPE_Length', 
                                           utility_lines.get('SHAPE__Length', 
                                           pd.Series([0]))).sum() / 5280, 2),
                        'avg_voltage': round(utility_lines['VOLTAGE'].replace(-999999, None).mean(), 2),
                        'voltage_classes': utility_lines['VOLT_CLASS'].nunique()
                    })
            
            return utilities
            
        except Exception as e:
            logger.error(f"Failed to get major utilities: {e}")
            return []
    
    def analyze_grid_connectivity(self, region: Optional[str] = None) -> Dict[str, Any]:
        """Analyze grid connectivity and identify critical corridors.
        
        Args:
            region: Optional region/state filter.
            
        Returns:
            Dictionary with connectivity analysis.
        """
        if self.lines_data is None or self.lines_data.empty:
            return {}
        
        try:
            df = self.lines_data
            
            # Filter by region if specified
            if region and 'STATE' in df.columns:
                df = df[df['STATE'].str.contains(region, case=False, na=False)]
            
            # Identify high-voltage backbone lines (345 kV and above)
            backbone_lines = df[df['VOLTAGE'] >= 345]
            
            # Find major interconnection corridors
            if 'SUB_1' in df.columns and 'SUB_2' in df.columns:
                # Count lines between substation pairs
                connections = df.groupby(['SUB_1', 'SUB_2']).size().reset_index(name='line_count')
                critical_corridors = connections.nlargest(10, 'line_count')
            else:
                critical_corridors = pd.DataFrame()
            
            analysis = {
                'total_lines': len(df),
                'backbone_lines': len(backbone_lines),
                'backbone_percentage': round(len(backbone_lines) / len(df) * 100, 2),
                'avg_backbone_voltage': round(backbone_lines['VOLTAGE'].mean(), 2),
                'unique_substations': len(set(list(df['SUB_1'].dropna().unique()) + list(df['SUB_2'].dropna().unique()))),
                'critical_corridors': critical_corridors.to_dict('records') if not critical_corridors.empty else []
            }
            
            # Analyze redundancy (parallel lines)
            if not critical_corridors.empty:
                analysis['max_parallel_lines'] = int(critical_corridors['line_count'].max())
                analysis['avg_parallel_lines'] = round(critical_corridors['line_count'].mean(), 2)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze connectivity: {e}")
            return {}
    
    def search_lines(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search transmission lines by multiple criteria.
        
        Args:
            query: Search query string.
            limit: Maximum results to return.
            
        Returns:
            List of matching transmission line records.
        """
        if self.lines_data is None or self.lines_data.empty:
            return []
        
        try:
            # Search across multiple columns
            mask = (
                self.lines_data['OWNER'].str.contains(query, case=False, na=False) |
                self.lines_data['SUB_1'].str.contains(query, case=False, na=False) |
                self.lines_data['SUB_2'].str.contains(query, case=False, na=False) |
                self.lines_data['VOLT_CLASS'].str.contains(query, case=False, na=False)
            )
            
            results = self.lines_data[mask].head(limit)
            return results.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to search transmission lines: {e}")
            return []
    
    def export_for_visualization(self, voltage_filter: Optional[str] = None, 
                                 limit: int = 5000) -> Dict[str, Any]:
        """Export data formatted for map visualization (GeoJSON compatible).
        
        Args:
            voltage_filter: Optional voltage class filter.
            limit: Maximum number of lines to export.
            
        Returns:
            Dictionary with GeoJSON-like structure.
        """
        if self.lines_data is None or self.lines_data.empty:
            return {"type": "FeatureCollection", "features": []}
        
        try:
            df = self.lines_data
            
            # Apply voltage filter
            if voltage_filter:
                df = df[df['VOLT_CLASS'] == voltage_filter]
            
            # Sample for performance
            if len(df) > limit:
                df = df.sample(n=limit)
            
            features = []
            for _, line in df.iterrows():
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": line.get('ID', ''),
                        "type": line.get('TYPE', ''),
                        "status": line.get('STATUS', ''),
                        "owner": line.get('OWNER', ''),
                        "voltage": line.get('VOLTAGE', ''),
                        "volt_class": line.get('VOLT_CLASS', ''),
                        "sub_1": line.get('SUB_1', ''),
                        "sub_2": line.get('SUB_2', ''),
                        "length_miles": round(line.get('SHAPE__Length', 0) / 5280, 2)
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": self._extract_coordinates(line)
                    }
                }
                features.append(feature)
            
            return {
                "type": "FeatureCollection",
                "features": features,
                "metadata": {
                    "total_available": len(self.lines_data),
                    "displayed": len(features),
                    "voltage_filter": voltage_filter
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to export for visualization: {e}")
            return {"type": "FeatureCollection", "features": []}
    
    def _extract_coordinates(self, line_row: pd.Series) -> List[List[float]]:
        """Extract coordinate pairs from line geometry."""
        # Simplified - in production, parse actual geometry fields
        coords = []
        
        if 'START_LON' in line_row and 'START_LAT' in line_row:
            start_lon = line_row['START_LON']
            start_lat = line_row['START_LAT']
            
            if pd.notna(start_lon) and pd.notna(start_lat):
                coords.append([float(start_lon), float(start_lat)])
        
        if 'END_LON' in line_row and 'END_LAT' in line_row:
            end_lon = line_row['END_LON']
            end_lat = line_row['END_LAT']
            
            if pd.notna(end_lon) and pd.notna(end_lat):
                coords.append([float(end_lon), float(end_lat)])
        
        return coords

# Example usage
service = TransmissionLinesService("HIFLD_Transmission_Lines.parquet")

# Get comprehensive statistics
stats = service.get_statistics()
print(f"Total Transmission Lines: {stats['total_lines']:,}")
print(f"Total Network Length: {stats['total_length_miles']:,.0f} miles")
print(f"Average Voltage: {stats['avg_voltage']:.0f} kV")
print(f"Unique Owners: {stats['unique_owners']:,}")

# Analyze major utilities
major_utilities = service.get_major_utilities(limit=10)
print("\nTop 10 Utilities by Line Count:")
for utility in major_utilities:
    print(f"  {utility['name'][:40]:40s} | {utility['line_count']:>6,} lines | {utility['total_miles']:>8,.0f} miles")

# Analyze grid connectivity
connectivity = service.analyze_grid_connectivity()
print(f"\nGrid Connectivity Analysis:")
print(f"  Backbone lines (>=345 kV): {connectivity['backbone_lines']:,} ({connectivity['backbone_percentage']:.1f}%)")
print(f"  Unique substations: {connectivity['unique_substations']:,}")
print(f"  Maximum parallel lines: {connectivity.get('max_parallel_lines', 'N/A')}")

# ======================================================================
# Code Block 2
# ======================================================================

def identify_critical_corridors(service: TransmissionLinesService, 
                               min_voltage: int = 345) -> pd.DataFrame:
    """Identify critical transmission corridors based on voltage and parallel lines.
    
    Args:
        service: Initialized TransmissionLinesService.
        min_voltage: Minimum voltage threshold for critical classification.
        
    Returns:
        DataFrame with critical corridor analysis.
    """
    df = service.lines_data
    
    # Filter for high-voltage lines
    critical_lines = df[df['VOLTAGE'] >= min_voltage].copy()
    
    # Group by substation pairs to find corridors
    corridors = critical_lines.groupby(['SUB_1', 'SUB_2']).agg({
        'ID': 'count',  # Number of parallel lines
        'VOLTAGE': ['mean', 'max'],
        'OWNER': lambda x: ', '.join(x.unique()[:3]),  # Top owners
        'SHAPE__Length': 'sum'
    }).reset_index()
    
    # Flatten column names
    corridors.columns = ['substation_1', 'substation_2', 'parallel_lines', 
                        'avg_voltage', 'max_voltage', 'owners', 'total_length']
    
    # Calculate criticality score (higher = more critical)
    # Factors: voltage, parallel lines (redundancy), length
    corridors['criticality_score'] = (
        (corridors['max_voltage'] / 765) * 0.4 +  # Voltage factor
        (corridors['parallel_lines'] / corridors['parallel_lines'].max()) * 0.4 +  # Redundancy
        (corridors['total_length'] / corridors['total_length'].max()) * 0.2  # Length
    ) * 100
    
    # Sort by criticality
    corridors = corridors.sort_values('criticality_score', ascending=False)
    
    return corridors

# Example usage
corridors_df = identify_critical_corridors(service, min_voltage=345)

print("Top 10 Critical Transmission Corridors:")
print(f"{'Substation 1':<30} {'Substation 2':<30} {'Lines':<6} {'Max kV':<8} {'Score':<6}")
print("-" * 90)

for _, corridor in corridors_df.head(10).iterrows():
    print(f"{str(corridor['substation_1'])[:28]:<30} "
          f"{str(corridor['substation_2'])[:28]:<30} "
          f"{corridor['parallel_lines']:<6} "
          f"{corridor['max_voltage']:<8.0f} "
          f"{corridor['criticality_score']:<6.1f}")

# ======================================================================
# Code Block 3
# ======================================================================

def analyze_voltage_hierarchy(service: TransmissionLinesService) -> Dict[str, Any]:
    """Analyze transmission grid by voltage class hierarchy.
    
    Returns:
        Dictionary with voltage class analysis.
    """
    df = service.lines_data
    
    # Define voltage class categories
    voltage_categories = {
        'Ultra-High Voltage (>=500 kV)': df[df['VOLTAGE'] >= 500],
        'Extra-High Voltage (345-499 kV)': df[(df['VOLTAGE'] >= 345) & (df['VOLTAGE'] < 500)],
        'High Voltage (220-344 kV)': df[(df['VOLTAGE'] >= 220) & (df['VOLTAGE'] < 345)],
        'Sub-Transmission (100-219 kV)': df[(df['VOLTAGE'] >= 100) & (df['VOLTAGE'] < 220)]
    }
    
    analysis = {}
    
    for category, lines in voltage_categories.items():
        if len(lines) > 0:
            analysis[category] = {
                'line_count': len(lines),
                'percentage': round(len(lines) / len(df) * 100, 2),
                'total_miles': round(lines.get('SHAPE__Length', pd.Series([0])).sum() / 5280, 2),
                'avg_voltage': round(lines['VOLTAGE'].mean(), 2),
                'unique_owners': lines['OWNER'].nunique(),
                'overhead_percentage': round(
                    len(lines[lines['TYPE'].str.contains('OVERHEAD', na=False)]) / len(lines) * 100, 2
                )
            }
    
    return analysis

# Run analysis
hierarchy = analyze_voltage_hierarchy(service)

print("Transmission Grid Voltage Hierarchy:")
print("=" * 90)

for category, stats in hierarchy.items():
    print(f"\n{category}")
    print(f"  Lines: {stats['line_count']:,} ({stats['percentage']:.1f}% of total)")
    print(f"  Network length: {stats['total_miles']:,.0f} miles")
    print(f"  Average voltage: {stats['avg_voltage']:.0f} kV")
    print(f"  Unique owners: {stats['unique_owners']}")
    print(f"  Overhead: {stats['overhead_percentage']:.1f}%")

# ======================================================================
# Code Block 4
# ======================================================================

def map_utility_territories(service: TransmissionLinesService, 
                           top_n: int = 15) -> pd.DataFrame:
    """Create utility territory profile based on transmission ownership.
    
    Args:
        service: Initialized TransmissionLinesService.
        top_n: Number of top utilities to analyze.
        
    Returns:
        DataFrame with utility territory analysis.
    """
    df = service.lines_data
    
    # Aggregate by owner
    utility_profile = df.groupby('OWNER').agg({
        'ID': 'count',
        'VOLTAGE': ['mean', 'max', 'min'],
        'SHAPE__Length': 'sum',
        'VOLT_CLASS': 'nunique',
        'STATUS': lambda x: (x == 'IN SERVICE').sum()
    }).reset_index()
    
    # Flatten columns
    utility_profile.columns = ['owner', 'total_lines', 'avg_voltage', 'max_voltage', 
                              'min_voltage', 'total_length', 'voltage_classes', 'in_service']
    
    # Calculate derived metrics
    utility_profile['total_miles'] = utility_profile['total_length'] / 5280
    utility_profile['in_service_pct'] = (utility_profile['in_service'] / utility_profile['total_lines'] * 100).round(2)
    
    # Filter out invalid owners
    utility_profile = utility_profile[
        ~utility_profile['owner'].isin(['NOT AVAILABLE', 'UNKNOWN', '-999999'])
    ]
    
    # Sort by total lines and select top N
    utility_profile = utility_profile.nlargest(top_n, 'total_lines')
    
    return utility_profile

# Generate utility territory map
territories = map_utility_territories(service, top_n=15)

print("Major Transmission Owners:")
print(f"{'Utility':<40} {'Lines':<8} {'Miles':<10} {'Avg kV':<8} {'In Service':<10}")
print("-" * 90)

for _, row in territories.iterrows():
    print(f"{str(row['owner'])[:38]:<40} "
          f"{row['total_lines']:<8,} "
          f"{row['total_miles']:<10,.0f} "
          f"{row['avg_voltage']:<8.0f} "
          f"{row['in_service_pct']:<10.1f}%")

# ======================================================================
# Code Block 5
# ======================================================================

def create_interactive_grid_map(service: TransmissionLinesService, 
                               output_file: str = "grid_map.geojson") -> str:
    """Create interactive grid map data for web visualization.
    
    Args:
        service: Initialized TransmissionLinesService.
        output_file: Output filename for GeoJSON.
        
    Returns:
        Path to generated GeoJSON file.
    """
    import json
    
    # Export high-voltage backbone (345 kV and above)
    geojson_data = service.export_for_visualization(
        voltage_filter="345-765",
        limit=10000
    )
    
    # Add additional metadata for visualization
    geojson_data['metadata']['map_config'] = {
        'center': [-98.5795, 39.8283],  # Geographic center of US
        'zoom': 4,
        'style': {
            '765 kV': {'color': '#C73E1D', 'weight': 4},  # Red - Ultra high
            '500 kV': {'color': '#F18F01', 'weight': 3},  # Orange - High
            '345 kV': {'color': '#2E86AB', 'weight': 2}   # Blue - Standard
        }
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"Generated interactive grid map: {output_file}")
    print(f"  Lines included: {len(geojson_data['features']):,}")
    print(f"  Total available: {geojson_data['metadata']['total_available']:,}")
    
    return output_file

# Create map
map_file = create_interactive_grid_map(service)

# ======================================================================
# Code Block 6
# ======================================================================

def analyze_corridor_capacity(service: TransmissionLinesService,
                              forecast_load_mw: float,
                              region_corridors: List[str]) -> Dict[str, Any]:
    """Analyze whether transmission corridors can handle forecast load.
    
    Args:
        service: TransmissionLinesService instance.
        forecast_load_mw: Forecast peak load in MW.
        region_corridors: List of corridor identifiers.
        
    Returns:
        Capacity analysis results.
    """
    df = service.lines_data
    
    # Typical line capacity by voltage (MW per line, simplified)
    voltage_capacity = {
        765: 2400,  # 765 kV can carry ~2400 MW
        500: 1200,
        345: 600,
        230: 300,
        138: 150
    }
    
    analysis = {
        'forecast_load_mw': forecast_load_mw,
        'corridors': [],
        'total_capacity_mw': 0,
        'utilization_pct': 0,
        'capacity_adequate': False
    }
    
    for corridor_id in region_corridors:
        # Find lines in this corridor
        corridor_lines = df[df['SUB_1'].str.contains(corridor_id, na=False) |
                           df['SUB_2'].str.contains(corridor_id, na=False)]
        
        if len(corridor_lines) > 0:
            # Calculate total corridor capacity
            corridor_capacity = 0
            for _, line in corridor_lines.iterrows():
                voltage = line['VOLTAGE']
                # Find closest voltage class
                closest_voltage = min(voltage_capacity.keys(), 
                                    key=lambda x: abs(x - voltage))
                corridor_capacity += voltage_capacity[closest_voltage]
            
            analysis['corridors'].append({
                'id': corridor_id,
                'line_count': len(corridor_lines),
                'capacity_mw': corridor_capacity,
                'avg_voltage': corridor_lines['VOLTAGE'].mean()
            })
            
            analysis['total_capacity_mw'] += corridor_capacity
    
    # Calculate utilization
    if analysis['total_capacity_mw'] > 0:
        analysis['utilization_pct'] = round(
            (forecast_load_mw / analysis['total_capacity_mw']) * 100, 2
        )
        analysis['capacity_adequate'] = analysis['utilization_pct'] < 80  # 80% threshold
    
    return analysis

# Example capacity analysis
forecast_mw = 28500  # Peak forecast load
region_corridors = ['SUBSTATION_A', 'SUBSTATION_B', 'SUBSTATION_C']

capacity = analyze_corridor_capacity(service, forecast_mw, region_corridors)

print(f"\nTransmission Capacity Analysis")
print(f"Forecast Peak Load: {capacity['forecast_load_mw']:,} MW")
print(f"Total Corridor Capacity: {capacity['total_capacity_mw']:,} MW")
print(f"Utilization: {capacity['utilization_pct']:.1f}%")
print(f"Capacity Adequate: {'Yes' if capacity['capacity_adequate'] else 'No'}")

print(f"\nCorridor Breakdown:")
for corridor in capacity['corridors']:
    print(f"  {corridor['id']}: {corridor['capacity_mw']:,} MW ({corridor['line_count']} lines)")

# ======================================================================
# Code Block 7
# ======================================================================

"""Service for handling transmission line data and grid visualization."""

def __init__(self, data_path: str = "HIFLD_Transmission_Lines.parquet"):
    """Initialize transmission lines service.
    
    Args:
        data_path: Path to the transmission lines data file.
    """
    self.data_path = data_path
    self.lines_data = None
    self._load_data()

def _load_data(self) -> None:
    """Load transmission lines data from Parquet or CSV."""
    try:
        if os.path.exists(self.data_path):
            logger.info(f"Loading transmission lines data from {self.data_path}")

# ======================================================================
# Code Block 8
# ======================================================================

if self.data_path.endswith('.parquet'):
                self.lines_data = pd.read_parquet(self.data_path)
            else:
                self.lines_data = pd.read_csv(self.data_path)
            
            logger.info(f"Loaded {len(self.lines_data):,} transmission lines")

# ======================================================================
# Code Block 9
# ======================================================================

if 'START_LAT' in self.lines_data.columns:
                self.lines_data['latitude'] = self.lines_data['START_LAT']
                self.lines_data['longitude'] = self.lines_data['START_LON']
            
        else:
            logger.warning(f"Transmission lines file not found: {self.data_path}")
            self.lines_data = pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Failed to load transmission lines data: {e}")
        self.lines_data = pd.DataFrame()

def get_statistics(self) -> Dict[str, Any]:
    """Get statistics about the transmission lines data."""
    if self.lines_data is None or self.lines_data.empty:
        return {}
    
    try:

# ======================================================================
# Code Block 10
# ======================================================================

stats = {
            "total_lines": len(self.lines_data),
            "in_service": len(self.lines_data[self.lines_data['STATUS'] == 'IN SERVICE']),
            "voltage_classes": len(self.lines_data['VOLT_CLASS'].dropna().unique()),
            "unique_owners": len(self.lines_data['OWNER'].dropna().unique()),
            "total_length_miles": self.lines_data.get('SHAPE_Length', 
                                                     self.lines_data.get('SHAPE__Length', 
                                                     pd.Series([0]))).sum() / 5280,
            "avg_voltage": self.lines_data['VOLTAGE'].replace(-999999, None).mean(),
            "overhead_lines": len(self.lines_data[self.lines_data['TYPE'].str.contains('OVERHEAD', na=False)]),
            "underground_lines": len(self.lines_data[self.lines_data['TYPE'].str.contains('UNDERGROUND', na=False)])
        }

# ======================================================================
# Code Block 11
# ======================================================================

voltage_distribution = self.get_voltage_class_distribution()
        stats['voltage_distribution'] = voltage_distribution

# ======================================================================
# Code Block 12
# ======================================================================

if 'STATE' in self.lines_data.columns:
            stats['states_covered'] = self.lines_data['STATE'].nunique()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to calculate statistics: {e}")
        return {}

def get_voltage_class_distribution(self) -> List[Dict[str, Any]]:
    """Get voltage class distribution with counts."""
    if self.lines_data is None or self.lines_data.empty:
        return []
    
    try:

# ======================================================================
# Code Block 13
# ======================================================================

valid_data = self.lines_data[
            (self.lines_data['VOLT_CLASS'].notna()) & 
            (self.lines_data['VOLT_CLASS'] != '-999999')
        ]
        
        voltage_counts = valid_data['VOLT_CLASS'].value_counts().head(10)
        
        return [
            {"voltage_class": str(vc), "count": int(count), "percentage": round(count/len(valid_data)*100, 2)}
            for vc, count in voltage_counts.items()
        ]
        
    except Exception as e:
        logger.error(f"Failed to get voltage distribution: {e}")
        return []

def get_lines_by_voltage_class(self, voltage_class: str = "345-765", limit: int = 1000) -> List[Dict[str, Any]]:
    """Get transmission lines filtered by voltage class.
    
    Args:
        voltage_class: Voltage class to filter by (e.g., '345-765', '220-287')
        limit: Maximum number of lines to return.
        
    Returns:
        List of transmission line records.
    """
    if self.lines_data is None or self.lines_data.empty:
        return []
    
    try:
        filtered_lines = self.lines_data[
            self.lines_data['VOLT_CLASS'] == voltage_class
        ].head(limit)
        
        return filtered_lines.to_dict('records')
        
    except Exception as e:
        logger.error(f"Failed to filter transmission lines: {e}")
        return []

def get_lines_by_owner(self, owner: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Get transmission lines filtered by owner/utility.
    
    Args:
        owner: Owner/utility name to filter by.
        limit: Maximum number of lines to return.
        
    Returns:
        List of transmission line records.
    """
    if self.lines_data is None or self.lines_data.empty:
        return []
    
    try:
        filtered_lines = self.lines_data[
            self.lines_data['OWNER'].str.contains(owner, case=False, na=False)
        ].head(limit)
        
        return filtered_lines.to_dict('records')
        
    except Exception as e:
        logger.error(f"Failed to filter by owner: {e}")
        return []

def get_major_utilities(self, limit: int = 20) -> List[Dict[str, Any]]:
    """Get major utilities ranked by transmission line count.
    
    Args:
        limit: Maximum number of utilities to return.
        
    Returns:
        List of utilities with line counts.
    """
    if self.lines_data is None or self.lines_data.empty:
        return []
    
    try:

# ======================================================================
# Code Block 14
# ======================================================================

utility_counts = self.lines_data['OWNER'].value_counts().head(limit)
        
        utilities = []
        for owner, count in utility_counts.items():
            if owner not in ['NOT AVAILABLE', 'UNKNOWN', '-999999']:

# ======================================================================
# Code Block 15
# ======================================================================

utility_lines = self.lines_data[self.lines_data['OWNER'] == owner]
                
                utilities.append({
                    'name': owner,
                    'line_count': int(count),
                    'total_miles': round(utility_lines.get('SHAPE_Length', 
                                       utility_lines.get('SHAPE__Length', 
                                       pd.Series([0]))).sum() / 5280, 2),
                    'avg_voltage': round(utility_lines['VOLTAGE'].replace(-999999, None).mean(), 2),
                    'voltage_classes': utility_lines['VOLT_CLASS'].nunique()
                })
        
        return utilities
        
    except Exception as e:
        logger.error(f"Failed to get major utilities: {e}")
        return []

def analyze_grid_connectivity(self, region: Optional[str] = None) -> Dict[str, Any]:
    """Analyze grid connectivity and identify critical corridors.
    
    Args:
        region: Optional region/state filter.
        
    Returns:
        Dictionary with connectivity analysis.
    """
    if self.lines_data is None or self.lines_data.empty:
        return {}
    
    try:
        df = self.lines_data

# ======================================================================
# Code Block 16
# ======================================================================

if region and 'STATE' in df.columns:
            df = df[df['STATE'].str.contains(region, case=False, na=False)]

# ======================================================================
# Code Block 17
# ======================================================================

backbone_lines = df[df['VOLTAGE'] >= 345]

# ======================================================================
# Code Block 18
# ======================================================================

connections = df.groupby(['SUB_1', 'SUB_2']).size().reset_index(name='line_count')
            critical_corridors = connections.nlargest(10, 'line_count')
        else:
            critical_corridors = pd.DataFrame()
        
        analysis = {
            'total_lines': len(df),
            'backbone_lines': len(backbone_lines),
            'backbone_percentage': round(len(backbone_lines) / len(df) * 100, 2),
            'avg_backbone_voltage': round(backbone_lines['VOLTAGE'].mean(), 2),
            'unique_substations': len(set(list(df['SUB_1'].dropna().unique()) + list(df['SUB_2'].dropna().unique()))),
            'critical_corridors': critical_corridors.to_dict('records') if not critical_corridors.empty else []
        }

# ======================================================================
# Code Block 19
# ======================================================================

if not critical_corridors.empty:
            analysis['max_parallel_lines'] = int(critical_corridors['line_count'].max())
            analysis['avg_parallel_lines'] = round(critical_corridors['line_count'].mean(), 2)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze connectivity: {e}")
        return {}

def search_lines(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Search transmission lines by multiple criteria.
    
    Args:
        query: Search query string.
        limit: Maximum results to return.
        
    Returns:
        List of matching transmission line records.
    """
    if self.lines_data is None or self.lines_data.empty:
        return []
    
    try:

# ======================================================================
# Code Block 20
# ======================================================================

mask = (
            self.lines_data['OWNER'].str.contains(query, case=False, na=False) |
            self.lines_data['SUB_1'].str.contains(query, case=False, na=False) |
            self.lines_data['SUB_2'].str.contains(query, case=False, na=False) |
            self.lines_data['VOLT_CLASS'].str.contains(query, case=False, na=False)
        )
        
        results = self.lines_data[mask].head(limit)
        return results.to_dict('records')
        
    except Exception as e:
        logger.error(f"Failed to search transmission lines: {e}")
        return []

def export_for_visualization(self, voltage_filter: Optional[str] = None, 
                             limit: int = 5000) -> Dict[str, Any]:
    """Export data formatted for map visualization (GeoJSON compatible).
    
    Args:
        voltage_filter: Optional voltage class filter.
        limit: Maximum number of lines to export.
        
    Returns:
        Dictionary with GeoJSON-like structure.
    """
    if self.lines_data is None or self.lines_data.empty:
        return {"type": "FeatureCollection", "features": []}
    
    try:
        df = self.lines_data

# ======================================================================
# Code Block 21
# ======================================================================

if voltage_filter:
            df = df[df['VOLT_CLASS'] == voltage_filter]

# ======================================================================
# Code Block 22
# ======================================================================

if len(df) > limit:
            df = df.sample(n=limit)
        
        features = []
        for _, line in df.iterrows():
            feature = {
                "type": "Feature",
                "properties": {
                    "id": line.get('ID', ''),
                    "type": line.get('TYPE', ''),
                    "status": line.get('STATUS', ''),
                    "owner": line.get('OWNER', ''),
                    "voltage": line.get('VOLTAGE', ''),
                    "volt_class": line.get('VOLT_CLASS', ''),
                    "sub_1": line.get('SUB_1', ''),
                    "sub_2": line.get('SUB_2', ''),
                    "length_miles": round(line.get('SHAPE__Length', 0) / 5280, 2)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": self._extract_coordinates(line)
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_available": len(self.lines_data),
                "displayed": len(features),
                "voltage_filter": voltage_filter
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to export for visualization: {e}")
        return {"type": "FeatureCollection", "features": []}

def _extract_coordinates(self, line_row: pd.Series) -> List[List[float]]:
    """Extract coordinate pairs from line geometry."""

# ======================================================================
# Code Block 23
# ======================================================================

coords = []
    
    if 'START_LON' in line_row and 'START_LAT' in line_row:
        start_lon = line_row['START_LON']
        start_lat = line_row['START_LAT']
        
        if pd.notna(start_lon) and pd.notna(start_lat):
            coords.append([float(start_lon), float(start_lat)])
    
    if 'END_LON' in line_row and 'END_LAT' in line_row:
        end_lon = line_row['END_LON']
        end_lat = line_row['END_LAT']
        
        if pd.notna(end_lon) and pd.notna(end_lat):
            coords.append([float(end_lon), float(end_lat)])
    
    return coords

# ======================================================================
# Code Block 24
# ======================================================================

print(f"  {utility['name'][:40]:40s} | {utility['line_count']:>6,} lines | {utility['total_miles']:>8,.0f} miles")

# ======================================================================
# Code Block 25
# ======================================================================

min_voltage: int = 345) -> pd.DataFrame:
"""Identify critical transmission corridors based on voltage and parallel lines.

Args:
    service: Initialized TransmissionLinesService.
    min_voltage: Minimum voltage threshold for critical classification.
    
Returns:
    DataFrame with critical corridor analysis.
"""
df = service.lines_data

# ======================================================================
# Code Block 26
# ======================================================================

critical_lines = df[df['VOLTAGE'] >= min_voltage].copy()

# ======================================================================
# Code Block 27
# ======================================================================

corridors = critical_lines.groupby(['SUB_1', 'SUB_2']).agg({
    'ID': 'count',  # Number of parallel lines
    'VOLTAGE': ['mean', 'max'],
    'OWNER': lambda x: ', '.join(x.unique()[:3]),  # Top owners
    'SHAPE__Length': 'sum'
}).reset_index()

# ======================================================================
# Code Block 28
# ======================================================================

corridors.columns = ['substation_1', 'substation_2', 'parallel_lines', 
                    'avg_voltage', 'max_voltage', 'owners', 'total_length']

# ======================================================================
# Code Block 29
# ======================================================================

corridors['criticality_score'] = (
    (corridors['max_voltage'] / 765) * 0.4 +  # Voltage factor
    (corridors['parallel_lines'] / corridors['parallel_lines'].max()) * 0.4 +  # Redundancy
    (corridors['total_length'] / corridors['total_length'].max()) * 0.2  # Length
) * 100

# ======================================================================
# Code Block 30
# ======================================================================

corridors = corridors.sort_values('criticality_score', ascending=False)

return corridors

# ======================================================================
# Code Block 31
# ======================================================================

print(f"{str(corridor['substation_1'])[:28]:<30} "
      f"{str(corridor['substation_2'])[:28]:<30} "
      f"{corridor['parallel_lines']:<6} "
      f"{corridor['max_voltage']:<8.0f} "
      f"{corridor['criticality_score']:<6.1f}")

# ======================================================================
# Code Block 32
# ======================================================================

"""Analyze transmission grid by voltage class hierarchy.

Returns:
    Dictionary with voltage class analysis.
"""
df = service.lines_data

# ======================================================================
# Code Block 33
# ======================================================================

voltage_categories = {
    'Ultra-High Voltage (>=500 kV)': df[df['VOLTAGE'] >= 500],
    'Extra-High Voltage (345-499 kV)': df[(df['VOLTAGE'] >= 345) & (df['VOLTAGE'] < 500)],
    'High Voltage (220-344 kV)': df[(df['VOLTAGE'] >= 220) & (df['VOLTAGE'] < 345)],
    'Sub-Transmission (100-219 kV)': df[(df['VOLTAGE'] >= 100) & (df['VOLTAGE'] < 220)]
}

analysis = {}

for category, lines in voltage_categories.items():
    if len(lines) > 0:
        analysis[category] = {
            'line_count': len(lines),
            'percentage': round(len(lines) / len(df) * 100, 2),
            'total_miles': round(lines.get('SHAPE__Length', pd.Series([0])).sum() / 5280, 2),
            'avg_voltage': round(lines['VOLTAGE'].mean(), 2),
            'unique_owners': lines['OWNER'].nunique(),
            'overhead_percentage': round(
                len(lines[lines['TYPE'].str.contains('OVERHEAD', na=False)]) / len(lines) * 100, 2
            )
        }

return analysis

# ======================================================================
# Code Block 34
# ======================================================================

print(f"\n{category}")
print(f"  Lines: {stats['line_count']:,} ({stats['percentage']:.1f}% of total)")
print(f"  Network length: {stats['total_miles']:,.0f} miles")
print(f"  Average voltage: {stats['avg_voltage']:.0f} kV")
print(f"  Unique owners: {stats['unique_owners']}")
print(f"  Overhead: {stats['overhead_percentage']:.1f}%")

# ======================================================================
# Code Block 35
# ======================================================================

top_n: int = 15) -> pd.DataFrame:
"""Create utility territory profile based on transmission ownership.

Args:
    service: Initialized TransmissionLinesService.
    top_n: Number of top utilities to analyze.
    
Returns:
    DataFrame with utility territory analysis.
"""
df = service.lines_data

# ======================================================================
# Code Block 36
# ======================================================================

utility_profile = df.groupby('OWNER').agg({
    'ID': 'count',
    'VOLTAGE': ['mean', 'max', 'min'],
    'SHAPE__Length': 'sum',
    'VOLT_CLASS': 'nunique',
    'STATUS': lambda x: (x == 'IN SERVICE').sum()
}).reset_index()

# ======================================================================
# Code Block 37
# ======================================================================

utility_profile.columns = ['owner', 'total_lines', 'avg_voltage', 'max_voltage', 
                          'min_voltage', 'total_length', 'voltage_classes', 'in_service']

# ======================================================================
# Code Block 38
# ======================================================================

utility_profile['total_miles'] = utility_profile['total_length'] / 5280
utility_profile['in_service_pct'] = (utility_profile['in_service'] / utility_profile['total_lines'] * 100).round(2)

# ======================================================================
# Code Block 39
# ======================================================================

utility_profile = utility_profile[
    ~utility_profile['owner'].isin(['NOT AVAILABLE', 'UNKNOWN', '-999999'])
]

# ======================================================================
# Code Block 40
# ======================================================================

utility_profile = utility_profile.nlargest(top_n, 'total_lines')

return utility_profile

# ======================================================================
# Code Block 41
# ======================================================================

print(f"{str(row['owner'])[:38]:<40} "
      f"{row['total_lines']:<8,} "
      f"{row['total_miles']:<10,.0f} "
      f"{row['avg_voltage']:<8.0f} "
      f"{row['in_service_pct']:<10.1f}%")

# ======================================================================
# Code Block 42
# ======================================================================

output_file: str = "grid_map.geojson") -> str:
"""Create interactive grid map data for web visualization.

Args:
    service: Initialized TransmissionLinesService.
    output_file: Output filename for GeoJSON.
    
Returns:
    Path to generated GeoJSON file.
"""
import json

# ======================================================================
# Code Block 43
# ======================================================================

geojson_data = service.export_for_visualization(
    voltage_filter="345-765",
    limit=10000
)

# ======================================================================
# Code Block 44
# ======================================================================

geojson_data['metadata']['map_config'] = {
    'center': [-98.5795, 39.8283],  # Geographic center of US
    'zoom': 4,
    'style': {
        '765 kV': {'color': '#C73E1D', 'weight': 4},  # Red - Ultra high
        '500 kV': {'color': '#F18F01', 'weight': 3},  # Orange - High
        '345 kV': {'color': '#2E86AB', 'weight': 2}   # Blue - Standard
    }
}

# ======================================================================
# Code Block 45
# ======================================================================

with open(output_file, 'w') as f:
    json.dump(geojson_data, f, indent=2)

print(f"Generated interactive grid map: {output_file}")
print(f"  Lines included: {len(geojson_data['features']):,}")
print(f"  Total available: {geojson_data['metadata']['total_available']:,}")

return output_file

# ======================================================================
# Code Block 46
# ======================================================================

forecast_load_mw: float,
                          region_corridors: List[str]) -> Dict[str, Any]:
"""Analyze whether transmission corridors can handle forecast load.

Args:
    service: TransmissionLinesService instance.
    forecast_load_mw: Forecast peak load in MW.
    region_corridors: List of corridor identifiers.
    
Returns:
    Capacity analysis results.
"""
df = service.lines_data

# ======================================================================
# Code Block 47
# ======================================================================

voltage_capacity = {
    765: 2400,  # 765 kV can carry ~2400 MW
    500: 1200,
    345: 600,
    230: 300,
    138: 150
}

analysis = {
    'forecast_load_mw': forecast_load_mw,
    'corridors': [],
    'total_capacity_mw': 0,
    'utilization_pct': 0,
    'capacity_adequate': False
}

for corridor_id in region_corridors:

# ======================================================================
# Code Block 48
# ======================================================================

corridor_lines = df[df['SUB_1'].str.contains(corridor_id, na=False) |
                       df['SUB_2'].str.contains(corridor_id, na=False)]
    
    if len(corridor_lines) > 0:

# ======================================================================
# Code Block 49
# ======================================================================

corridor_capacity = 0
        for _, line in corridor_lines.iterrows():
            voltage = line['VOLTAGE']

# ======================================================================
# Code Block 50
# ======================================================================

closest_voltage = min(voltage_capacity.keys(), 
                                key=lambda x: abs(x - voltage))
            corridor_capacity += voltage_capacity[closest_voltage]
        
        analysis['corridors'].append({
            'id': corridor_id,
            'line_count': len(corridor_lines),
            'capacity_mw': corridor_capacity,
            'avg_voltage': corridor_lines['VOLTAGE'].mean()
        })
        
        analysis['total_capacity_mw'] += corridor_capacity

# ======================================================================
# Code Block 51
# ======================================================================

if analysis['total_capacity_mw'] > 0:
    analysis['utilization_pct'] = round(
        (forecast_load_mw / analysis['total_capacity_mw']) * 100, 2
    )
    analysis['capacity_adequate'] = analysis['utilization_pct'] < 80  # 80% threshold

return analysis

# ======================================================================
# Code Block 52
# ======================================================================

print(f"  {corridor['id']}: {corridor['capacity_mw']:,} MW ({corridor['line_count']} lines)")
