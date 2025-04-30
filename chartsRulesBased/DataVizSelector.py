import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import warnings
import re
import os

class DataVizSelector:
    """
    A class that analyzes a pandas DataFrame and suggests/creates
    appropriate visualizations using Plotly.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataVizSelector with a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to analyze
        """
        self.df = df.copy()
        self.num_rows = df.shape[0]
        self.num_cols = df.shape[1]
        
        # Analyze column types
        self._analyze_column_types()
        
        # Generate suggestions dictionary
        self.suggestions = self._generate_suggestions()
    
    def _analyze_column_types(self):
        """Analyze and categorize columns in the DataFrame."""
        # Convert string date columns to datetime if possible
        for col in self.df.select_dtypes(include=['object']).columns:
            try:
                # Check if column might be a date
                if self.df[col].str.match(r'^\d{4}-\d{2}-\d{2}').any() or \
                   self.df[col].str.match(r'^\d{1,2}/\d{1,2}/\d{2,4}').any():
                    self.df[col] = pd.to_datetime(self.df[col], errors='ignore')
            except:
                pass
        
        # Get column types
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.numeric_cols = self.df.select_dtypes(include=['int', 'float']).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Simple column counts
        self.num_categorical = len(self.categorical_cols)
        self.num_numeric = len(self.numeric_cols)
        self.num_datetime = len(self.datetime_cols)
        
        # Further analyze categorical columns
        self.categorical_cardinality = {col: self.df[col].nunique() for col in self.categorical_cols}
        
        # Check for potential ID columns (high cardinality categorical or numeric)
        self.potential_id_cols = []
        for col in self.categorical_cols:
            if self.categorical_cardinality[col] > 0.9 * self.num_rows:
                self.potential_id_cols.append(col)
        
        # Check for binary columns
        self.binary_cols = [col for col in self.categorical_cols 
                          if self.categorical_cardinality[col] == 2]
        
        # Check for geospatial data hints
        self.geo_columns = [col for col in self.df.columns 
                          if any(geo_term in col.lower() 
                                for geo_term in ['country', 'state', 'city', 
                                                'latitude', 'longitude', 'lat', 'lon', 
                                                'county', 'region', 'postal', 'zip'])]
                                                
        # Identify latitude/longitude pairs
        self.has_lat_lon = bool(set(col.lower() for col in self.geo_columns) & 
                               set(['latitude', 'longitude', 'lat', 'lon']))
    
    def _generate_suggestions(self) -> Dict[str, Dict]:
        """
        Generate visualization suggestions based on data characteristics.
        
        Returns:
        --------
        dict : Dictionary of visualization suggestions with details
        """
        suggestions = {}
        
        # ===== Distribution visualizations =====
        if self.num_numeric >= 1:
            for col in self.numeric_cols[:3]:  # Limit to first 3 columns for brevity
                suggestions[f"histogram_{col}"] = {
                    "title": f"Histogram of {col}",
                    "description": f"Shows distribution of {col} values",
                    "viz_type": "histogram",
                    "columns": [col],
                    "complexity": "simple"
                }
                
                suggestions[f"box_{col}"] = {
                    "title": f"Box Plot of {col}",
                    "description": f"Shows statistical summary of {col} distribution",
                    "viz_type": "box",
                    "columns": [col],
                    "complexity": "simple"
                }
        
        # ===== Categorical visualizations =====
        if self.num_categorical >= 1:
            for col in self.categorical_cols[:3]:  # Limit to first 3 columns
                if self.categorical_cardinality[col] <= 20:  # Only if reasonable number of categories
                    suggestions[f"bar_{col}"] = {
                        "title": f"Bar Chart of {col}",
                        "description": f"Shows frequency of each {col} category",
                        "viz_type": "bar",
                        "columns": [col],
                        "complexity": "simple"
                    }
                    
                    if self.categorical_cardinality[col] <= 10:
                        suggestions[f"pie_{col}"] = {
                            "title": f"Pie Chart of {col}",
                            "description": f"Shows proportion of each {col} category",
                            "viz_type": "pie",
                            "columns": [col],
                            "complexity": "simple"
                        }
        
        # ===== Relationships between categorical and numeric =====
        if self.num_categorical >= 1 and self.num_numeric >= 1:
            for cat_col in self.categorical_cols[:2]:  # Limit combinations
                if self.categorical_cardinality[cat_col] <= 20:  # Only if reasonable
                    for num_col in self.numeric_cols[:2]:
                        suggestions[f"boxplot_{cat_col}_{num_col}"] = {
                            "title": f"{num_col} by {cat_col}",
                            "description": f"Compare distribution of {num_col} across {cat_col} categories",
                            "viz_type": "box_by_category",
                            "columns": [cat_col, num_col],
                            "complexity": "medium"
                        }
                        
                        suggestions[f"barplot_{cat_col}_{num_col}"] = {
                            "title": f"Average {num_col} by {cat_col}",
                            "description": f"Compare average {num_col} across {cat_col} categories",
                            "viz_type": "bar_by_category",
                            "columns": [cat_col, num_col],
                            "complexity": "medium"
                        }
        
        # ===== Correlations between numeric variables =====
        if self.num_numeric >= 2:
            for i, col1 in enumerate(self.numeric_cols[:3]):
                for col2 in self.numeric_cols[i+1:4]:  # Avoid duplicates
                    suggestions[f"scatter_{col1}_{col2}"] = {
                        "title": f"Scatter Plot: {col1} vs {col2}",
                        "description": f"Examine relationship between {col1} and {col2}",
                        "viz_type": "scatter",
                        "columns": [col1, col2],
                        "complexity": "medium"
                    }
            
            if self.num_numeric >= 3:
                suggestions["heatmap_correlation"] = {
                    "title": "Correlation Heatmap",
                    "description": "Visualize correlations between numeric variables",
                    "viz_type": "correlation_heatmap",
                    "columns": self.numeric_cols,
                    "complexity": "advanced"
                }
        
        # ===== Time series visualizations =====
        if self.num_datetime >= 1 and self.num_numeric >= 1:
            time_col = self.datetime_cols[0]
            for num_col in self.numeric_cols[:3]:
                suggestions[f"line_{time_col}_{num_col}"] = {
                    "title": f"{num_col} Over Time",
                    "description": f"Track changes in {num_col} over {time_col}",
                    "viz_type": "time_series",
                    "columns": [time_col, num_col],
                    "complexity": "medium"
                }
            
            if self.num_numeric >= 2:
                suggestions["multi_line_time_series"] = {
                    "title": "Multiple Metrics Over Time",
                    "description": f"Compare multiple metrics over {time_col}",
                    "viz_type": "multi_time_series",
                    "columns": [time_col] + self.numeric_cols[:3],
                    "complexity": "advanced"
                }
        
        # ===== Geospatial visualizations =====
        if self.has_lat_lon:
            lat_col = next((col for col in self.df.columns if col.lower() in ['latitude', 'lat']), None)
            lon_col = next((col for col in self.df.columns if col.lower() in ['longitude', 'lon']), None)
            
            if lat_col and lon_col:
                suggestions["scatter_geo"] = {
                    "title": "Geographic Scatter Plot",
                    "description": "Plot points on a map",
                    "viz_type": "scatter_geo",
                    "columns": [lat_col, lon_col],
                    "complexity": "advanced"
                }
                
                if self.num_numeric >= 1:
                    for num_col in self.numeric_cols[:2]:
                        suggestions[f"scatter_geo_{num_col}"] = {
                            "title": f"Geographic Bubble Chart with {num_col}",
                            "description": f"Plot points on a map with size based on {num_col}",
                            "viz_type": "bubble_geo",
                            "columns": [lat_col, lon_col, num_col],
                            "complexity": "advanced"
                        }
        
        # ===== Multi-categorical visualizations =====
        if self.num_categorical >= 2:
            cat_cols_subset = [col for col in self.categorical_cols 
                             if self.categorical_cardinality[col] <= 20][:3]
            
            if len(cat_cols_subset) >= 2:
                suggestions["heatmap_categorical"] = {
                    "title": f"Heatmap of {cat_cols_subset[0]} vs {cat_cols_subset[1]}",
                    "description": "Show frequency of combinations of categories",
                    "viz_type": "categorical_heatmap",
                    "columns": cat_cols_subset[:2],
                    "complexity": "advanced"
                }
                
                if self.num_numeric >= 1:
                    suggestions["grouped_bar"] = {
                        "title": f"Grouped Bar Chart",
                        "description": f"Compare {self.numeric_cols[0]} across multiple categories",
                        "viz_type": "grouped_bar",
                        "columns": cat_cols_subset[:2] + [self.numeric_cols[0]],
                        "complexity": "advanced"
                    }
        
        # ===== Multi-dimensional visualizations =====
        if self.num_numeric >= 3:
            if self.num_categorical >= 1 and self.categorical_cardinality[self.categorical_cols[0]] <= 10:
                suggestions["scatter_3d"] = {
                    "title": "3D Scatter Plot",
                    "description": "Visualize relationship between three numeric variables",
                    "viz_type": "scatter_3d",
                    "columns": self.numeric_cols[:3] + [self.categorical_cols[0]],
                    "complexity": "advanced"
                }
        
        return suggestions
    
    def get_suggestions(self) -> Dict[str, Dict]:
        """Return the visualization suggestions."""
        return self.suggestions
    
    def generate_visualization(self, viz_id: str) -> go.Figure:
        """
        Generate the specified visualization.
        
        Parameters:
        -----------
        viz_id : str
            The ID of the visualization to generate
            
        Returns:
        --------
        plotly.graph_objects.Figure
            The generated visualization
        """
        if viz_id not in self.suggestions:
            raise ValueError(f"Visualization ID '{viz_id}' not found in suggestions")
        
        viz_info = self.suggestions[viz_id]
        viz_type = viz_info["viz_type"]
        columns = viz_info["columns"]
        
        # Simple distribution visualizations
        if viz_type == "histogram":
            fig = px.histogram(self.df, x=columns[0], title=viz_info["title"])
            fig.update_layout(xaxis_title=columns[0], yaxis_title="Count")
        
        elif viz_type == "box":
            fig = px.box(self.df, y=columns[0], title=viz_info["title"])
            fig.update_layout(yaxis_title=columns[0])
        
        # Categorical visualizations
        elif viz_type == "bar":
            count_df = self.df[columns[0]].value_counts().reset_index()
            count_df.columns = [columns[0], 'Count']
            fig = px.bar(count_df, x=columns[0], y='Count', title=viz_info["title"])
            fig.update_layout(xaxis_title=columns[0], yaxis_title="Count")
        
        elif viz_type == "pie":
            count_df = self.df[columns[0]].value_counts().reset_index()
            count_df.columns = [columns[0], 'Count']
            fig = px.pie(count_df, names=columns[0], values='Count', title=viz_info["title"])
        
        # Relationship visualizations
        elif viz_type == "box_by_category":
            cat_col, num_col = columns
            fig = px.box(self.df, x=cat_col, y=num_col, title=viz_info["title"])
            fig.update_layout(xaxis_title=cat_col, yaxis_title=num_col)
        
        elif viz_type == "bar_by_category":
            cat_col, num_col = columns
            agg_df = self.df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(agg_df, x=cat_col, y=num_col, title=viz_info["title"])
            fig.update_layout(xaxis_title=cat_col, yaxis_title=f"Average {num_col}")
        
        elif viz_type == "scatter":
            col1, col2 = columns
            fig = px.scatter(self.df, x=col1, y=col2, title=viz_info["title"])
            fig.update_layout(xaxis_title=col1, yaxis_title=col2)
            
            # Add trendline
            fig.add_trace(go.Scatter(
                x=self.df[col1],
                y=self.df[col1].to_numpy() * np.polyfit(
                    self.df[col1].to_numpy(), self.df[col2].to_numpy(), 1)[0] + 
                    np.polyfit(self.df[col1].to_numpy(), self.df[col2].to_numpy(), 1)[1],
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
        
        elif viz_type == "correlation_heatmap":
            corr_matrix = self.df[columns].corr()
            fig = px.imshow(
                corr_matrix,
                title=viz_info["title"],
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig.update_layout(height=600, width=600)
        
        # Time series visualizations
        elif viz_type == "time_series":
            time_col, num_col = columns
            fig = px.line(self.df.sort_values(time_col), x=time_col, y=num_col, title=viz_info["title"])
            fig.update_layout(xaxis_title=time_col, yaxis_title=num_col)
        
        elif viz_type == "multi_time_series":
            time_col = columns[0]
            num_cols = columns[1:]
            
            fig = go.Figure()
            for col in num_cols:
                fig.add_trace(go.Scatter(
                    x=self.df[time_col],
                    y=self.df[col],
                    mode='lines',
                    name=col
                ))
            
            fig.update_layout(
                title=viz_info["title"],
                xaxis_title=time_col,
                yaxis_title="Value",
                legend_title="Metric"
            )
        
        # Geospatial visualizations
        elif viz_type == "scatter_geo":
            lat_col, lon_col = columns
            fig = px.scatter_mapbox(
                self.df, 
                lat=lat_col, 
                lon=lon_col,
                title=viz_info["title"],
                mapbox_style="open-street-map"
            )
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        
        elif viz_type == "bubble_geo":
            lat_col, lon_col, size_col = columns
            fig = px.scatter_mapbox(
                self.df, 
                lat=lat_col, 
                lon=lon_col,
                size=size_col,
                title=viz_info["title"],
                mapbox_style="open-street-map"
            )
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        
        # Multi-categorical visualizations
        elif viz_type == "categorical_heatmap":
            cat1, cat2 = columns
            crosstab = pd.crosstab(self.df[cat1], self.df[cat2])
            fig = px.imshow(
                crosstab,
                title=viz_info["title"],
                labels=dict(x=cat2, y=cat1, color="Count")
            )
            fig.update_layout(height=600, width=600)
        
        elif viz_type == "grouped_bar":
            cat1, cat2, num_col = columns
            agg_df = self.df.groupby([cat1, cat2])[num_col].mean().reset_index()
            fig = px.bar(
                agg_df, 
                x=cat1, 
                y=num_col, 
                color=cat2,
                barmode='group',
                title=viz_info["title"]
            )
            fig.update_layout(
                xaxis_title=cat1,
                yaxis_title=f"Average {num_col}",
                legend_title=cat2
            )
        
        # Multi-dimensional visualizations
        elif viz_type == "scatter_3d":
            x_col, y_col, z_col, color_col = columns
            fig = px.scatter_3d(
                self.df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_col,
                title=viz_info["title"]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col
                )
            )
        
        else:
            raise ValueError(f"Visualization type '{viz_type}' not implemented")
        
        return fig
    
    def generate_all_visualizations(self, max_viz: int = 10) -> Dict[str, go.Figure]:
        """
        Generate all suggested visualizations up to a maximum number.
        
        Parameters:
        -----------
        max_viz : int
            Maximum number of visualizations to generate
            
        Returns:
        --------
        dict : Dictionary of visualization IDs and their figures
        """
        visualizations = {}
        for i, (viz_id, viz_info) in enumerate(self.suggestions.items()):
            if i >= max_viz:
                break
            
            try:
                fig = self.generate_visualization(viz_id)
                visualizations[viz_id] = fig
            except Exception as e:
                warnings.warn(f"Failed to generate visualization '{viz_id}': {str(e)}")
        
        return visualizations
    
    def get_recommended_visualizations(self, top_n: int = 5) -> List[str]:
        """
        Get the top recommended visualizations based on data characteristics.
        
        Parameters:
        -----------
        top_n : int
            Number of recommendations to return
            
        Returns:
        --------
        list : List of visualization IDs
        """
        # Simple scoring system based on visualization complexity and data characteristics
        scores = {}
        
        for viz_id, viz_info in self.suggestions.items():
            score = 0
            
            # Base score by complexity
            complexity_scores = {"simple": 1, "medium": 2, "advanced": 3}
            score += complexity_scores.get(viz_info["complexity"], 0)
            
            # Boost score for visualizations that use more columns (more informative)
            score += len(viz_info["columns"]) * 0.5
            
            # Penalize visualizations with too many categories
            if viz_info["viz_type"] in ["bar", "pie"]:
                col = viz_info["columns"][0]
                if col in self.categorical_cols:
                    if self.categorical_cardinality[col] > 10:
                        score -= 1
                    if self.categorical_cardinality[col] > 20:
                        score -= 2
            
            # Boost time series visualizations if we have datetime data
            if viz_info["viz_type"] in ["time_series", "multi_time_series"] and self.num_datetime > 0:
                score += 1
            
            # Boost map visualizations if we have geographic data
            if viz_info["viz_type"] in ["scatter_geo", "bubble_geo"] and self.has_lat_lon:
                score += 2
                
            # Give extra weight to correlation visualizations with many numeric variables
            if viz_info["viz_type"] == "correlation_heatmap" and self.num_numeric > 3:
                score += 1
                
            scores[viz_id] = score
            
        # Sort by score and return top N
        top_viz = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [viz_id for viz_id, _ in top_viz]
    
    def describe_data(self) -> Dict:
        """
        Generate a summary description of the dataset.
        
        Returns:
        --------
        dict : Summary statistics and characteristics
        """
        summary = {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "column_types": {
                "categorical": self.categorical_cols,
                "numeric": self.numeric_cols,
                "datetime": self.datetime_cols
            },
            "categorical_cardinality": self.categorical_cardinality,
            "potential_id_cols": self.potential_id_cols,
            "binary_cols": self.binary_cols,
            "geo_columns": self.geo_columns,
            "has_lat_lon": self.has_lat_lon
        }
        
        # Add basic descriptive statistics for numeric columns
        if self.num_numeric > 0:
            summary["numeric_stats"] = self.df[self.numeric_cols].describe().to_dict()
            
        return summary


def run_interactive_visualization_selector():
    """
    Run the data visualization selector interactively, allowing the user
    to choose which dataset to analyze and which visualizations to display.
    """
    # List of available CSV files
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        print("Please make sure you have CSV files in the same directory as this script.")
        return
    
    print("Available datasets:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")
    
    # Get user input for dataset selection
    while True:
        try:
            choice = int(input("\nSelect a dataset (enter number): "))
            if 1 <= choice <= len(csv_files):
                selected_file = csv_files[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nLoading dataset: {selected_file}")
    df = pd.read_csv(selected_file)
    
    # Print dataset overview
    print(f"\nDataset overview:")
    print(f"- {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"- Column names: {', '.join(df.columns.tolist())}")
    
    # Initialize the visualization selector
    viz_selector = DataVizSelector(df)
    
    # Print data description
    description = viz_selector.describe_data()
    print("\nData Characteristics:")
    print(f"- {len(description['column_types']['categorical'])} categorical columns: {', '.join(description['column_types']['categorical'])}")
    print(f"- {len(description['column_types']['numeric'])} numeric columns: {', '.join(description['column_types']['numeric'])}")
    print(f"- {len(description['column_types']['datetime'])} datetime columns: {', '.join(description['column_types']['datetime'])}")
    
    # Get top visualization recommendations
    recommended_viz = viz_selector.get_recommended_visualizations(top_n=10)
    
    print("\nTop Recommended Visualizations:")
    for i, viz_id in enumerate(recommended_viz):
        viz_info = viz_selector.suggestions[viz_id]
        print(f"{i+1}. {viz_info['title']}: {viz_info['description']}")
    
    # Get user input for visualization selection
    while True:
        try:
            choice = int(input("\nSelect a visualization to display (enter number): "))
            if 1 <= choice <= len(recommended_viz):
                selected_viz = recommended_viz[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(recommended_viz)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Generate and display the selected visualization
    viz_info = viz_selector.suggestions[selected_viz]
    print(f"\nGenerating: {viz_info['title']}")
    fig = viz_selector.generate_visualization(selected_viz)
    
    # Save the visualization to HTML
    output_file = f"{selected_viz}_visualization.html"
    fig.write_html(output_file)
    print(f"\nVisualization saved to {output_file}")
    
    # Try to open the visualization in a browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(output_file))
        print("Opening visualization in your default web browser...")
    except:
        print(f"Please open {output_file} in your web browser to view the visualization")
    
    # Ask if the user wants to see another visualization
    choice = input("\nWould you like to see another visualization? (y/n): ")
    if choice.lower() in ['y', 'yes']:
        while True:
            try:
                choice = int(input("\nSelect another visualization to display (enter number): "))
                if 1 <= choice <= len(recommended_viz):
                    selected_viz = recommended_viz[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(recommended_viz)}")
            except ValueError:
                print("Please enter a valid number")
        
        viz_info = viz_selector.suggestions[selected_viz]
        print(f"\nGenerating: {viz_info['title']}")
        fig = viz_selector.generate_visualization(selected_viz)
        
        # Save the visualization to HTML
        output_file = f"{selected_viz}_visualization.html"
        fig.write_html(output_file)
        print(f"\nVisualization saved to {output_file}")
        
        # Try to open the visualization in a browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.realpath(output_file))
            print("Opening visualization in your default web browser...")
        except:
            print(f"Please open {output_file} in your web browser to view the visualization")

# Run the interactive selector if script is run directly
if __name__ == "__main__":
    run_interactive_visualization_selector()
