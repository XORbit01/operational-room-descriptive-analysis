"""
Reusable chart functions using Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb33',
    'info': '#17a2b8',
    'light': '#e0e0e0',
    'dark': '#333333'
}

COLOR_SEQUENCE = px.colors.qualitative.Set3

def pie_chart(df, values_col, title, color_map=None):
    """Create a pie chart"""
    value_counts = df[values_col].value_counts()
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title=title,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    return fig

def bar_chart(df, x_col, y_col=None, title="", orientation='v', color_col=None):
    """Create a bar chart"""
    if y_col is None:
        # Count values
        value_counts = df[x_col].value_counts()
        fig = px.bar(
            x=value_counts.index if orientation == 'v' else value_counts.values,
            y=value_counts.values if orientation == 'v' else value_counts.index,
            title=title,
            orientation='h' if orientation == 'h' else 'v',
            color_discrete_sequence=[COLORS['primary']]
        )
    else:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color=color_col,
            color_discrete_sequence=COLOR_SEQUENCE
        )
    fig.update_layout(height=400, showlegend=color_col is not None)
    return fig

def histogram(df, col, title="", bins=30, color_col=None):
    """Create a histogram"""
    fig = px.histogram(
        df,
        x=col,
        title=title,
        nbins=bins,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(height=400, showlegend=color_col is not None)
    return fig

def box_plot(df, y_col, x_col=None, title="", color_col=None):
    """Create a box plot"""
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(height=400, showlegend=color_col is not None)
    return fig

def scatter_plot(df, x_col, y_col, title="", color_col=None, size_col=None):
    """Create a scatter plot"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col,
        size=size_col,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(height=400, showlegend=color_col is not None)
    return fig

def line_chart(df, x_col, y_col, title="", color_col=None):
    """Create a line chart"""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(height=400, showlegend=color_col is not None)
    return fig

def heatmap(df, title=""):
    """Create a correlation heatmap"""
    corr = df.corr()
    fig = px.imshow(
        corr,
        title=title,
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=500)
    return fig

def gauge_chart(value, max_value, title, threshold=None):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': COLORS['success']},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': COLORS['warning']},
                {'range': [max_value * 0.8, max_value], 'color': COLORS['danger']}
            ] if threshold is None else [
                {'range': [0, threshold], 'color': COLORS['success']},
                {'range': [threshold, max_value], 'color': COLORS['danger']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold if threshold else max_value * 0.8
            }
        }
    ))
    fig.update_layout(height=300)
    return fig
