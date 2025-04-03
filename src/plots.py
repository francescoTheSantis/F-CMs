# plotly imports
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default="browser"    # or 'browser'
pio.templates.default="plotly_white"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import torch
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from plotly.subplots import make_subplots

# colors in rgb format
# 'cem' is red, 'blackbox' is grey, 'crm' is green, 'cbm_linear' is light_blue, 'cbm_mlp' is darker_blue
colors = {'cem': '255, 0, 0', 
          'blackbox': '128, 128, 128', 
          'cbm_linear': '64, 224, 208',
          'cbm_mlp': '0, 150, 255',
          'crm': '0, 128, 0'}

legend = {'blackbox': 'OpaqNN',
          'cem': 'CEM',
          'cbm_linear': 'CBM₊ₗᵢₙ',
          'cbm_mlp': 'CBM₊ₘₗₚ',
          'crm': 'C²BM'
}
markers = {'blackbox': 'circle',
           'cem': 'square',
           'cbm_linear': 'diamond',
           'cbm_mlp': 'cross',
           'crm': 'triangle-up'
    }
 
title_font_size = 80 # 80
axis_title_font_size = 65 # 70
legend_font_size = 22  # 22
tick_font_size = 53   # 57


def plot_intervention(y, y_std, title):
    # plotly histogram: concept names on x-axis, delta y accuracy on y-axis for each model
    model_names = list(y.keys())
    label_names = list(y[model_names[-1]].keys())

    # Create figure with secondary y-axis
    fig = go.Figure()
    for model_name in model_names:
        if y[model_name]:
            y_delta = list(y[model_name].values())
            y_std_data = list(y_std[model_name].values())
            fig.add_trace(go.Bar(x=label_names, 
                                 y=y_delta, 
                                 error_y=dict(type='data', array=y_std_data),
                                 name=legend[model_name],
                                 marker_color='rgb('+colors[model_name]+')'
                                 )
                        )
    # show all ticks
    fig.update_xaxes(tickmode='array', tickvals=list(label_names), ticktext=list(label_names))
    fig.update_layout(title=title,
                      title_x=0.5,
                      title_font_size=title_font_size,
                      xaxis_title='Intervened concept',
                      xaxis_title_font_size=axis_title_font_size,
                      yaxis_title="Rel. improv. (%) on task acc.",
                      yaxis_title_font_size=axis_title_font_size)
    # place legend at the bottom of the plot
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.97,
        xanchor="right",
        x=1
    ))
    # text size and tiks size
    fig.update_layout(font=dict(size=legend_font_size))
    fig.update_xaxes(tickfont=dict(size=tick_font_size))
    fig.update_yaxes(tickfont=dict(size=tick_font_size))
    # fig.show()
    return fig

def plot_level_intervention(y, y_std, y_label, title):
    # plotly histogram: concept names on x-axis, delta y accuracy on y-axis for each model
    model_names = list(y.keys())

    fig = go.Figure()
    y_delta = {}
    for model_name in model_names:
        if y[model_name]:
            # reordering the x-axis labels
            y_delta = {i: y[model_name][i] for i in range(len(y[model_name]))}
            y_std_data = {i: y_std[model_name][i] for i in range(len(y_std[model_name]))}
            x = list(y_delta.keys())
            # fill between the upper and lower bounds
            y_upper = list({i: y_delta[i] + y_std_data[i] for i in y_delta.keys()}.values())
            y_lower = list({i: y_delta[i] - y_std_data[i] for i in y_delta.keys()}.values())
            fig.add_trace(go.Scatter(x=x+x[::-1], # x, then x reversed
                                     y=y_upper+y_lower[::-1], # upper, then lower reversed
                                     fill='toself',
                                     line=dict(color='rgba('+colors[model_name]+',0.)'),
                                     fillcolor='rgba('+colors[model_name]+'0.05)',
                                     hoverinfo="skip",
                                     opacity=0.6,
                                     showlegend=False)
                        )
            # line plot with dots at the points
            fig.add_trace(go.Scatter(x=x, 
                                     y=list(y_delta.values()), 
                                     mode='lines+markers', 
                                     name=legend[model_name],
                                     marker=dict(size=25,
                                                 symbol=markers[model_name],
                                                 line=dict(width=6,
                                                           color='rgb('+colors[model_name]+')')
                                                 ), 
                                     line=dict(color='rgb('+colors[model_name]+')')
                                     )
                        )
    # update line size
    fig.update_traces(line=dict(width=5))

    # show all ticks
    fig.update_xaxes(tickmode='array', tickvals=list(y_delta.keys()), ticktext=list(y_delta.keys()))
    fig.update_layout(title=title,
                      title_x=0.5,
                      title_y=0.97,
                      title_font_size=title_font_size,
                      xaxis_title='Intervened level',
                      xaxis_title_font_size=axis_title_font_size,
                      yaxis_title=y_label,
                      yaxis_title_font_size=axis_title_font_size)
    # place legend at the bottom of the plot
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.93,
        xanchor="right",
        x=1
    ))
    # hide legend
    fig.update_layout(showlegend=False)
    fig.update_layout(font=dict(size=legend_font_size))
    fig.update_xaxes(tickfont=dict(size=tick_font_size))
    fig.update_yaxes(tickfont=dict(size=tick_font_size))
    return fig


def convert_adjMatrix_to_causallearnGraph(adj_matrix, node_labels):
    """
    Convert an adjacency matrix to a causallearn GeneralGraph.

    Args:
    adj_matrix (np.ndarray): An adjacency matrix representing a causal graph.
    node_labels (list): A list of node labels.

    Returns:
    GeneralGraph: A GeneralGraph object representing the graph.
    edges: a list of Edges object representing the edges in the graph.
    """
    # Initialize the GeneralGraph with nodes
    nodes = []
    for i in range(len(node_labels)):
        nodes.append(GraphNode(node_labels[i]))

    g_causal = GeneralGraph(nodes)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):  # Only check the upper triangle to avoid redundant checks
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                # Add undirected edge
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)
            elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
                # Add directed edge i → j
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)
            elif adj_matrix[i, j] == 0 and adj_matrix[j, i] == 1:
                # Add directed edge j → i
                edge = Edge(nodes[j], nodes[i], Endpoint.TAIL, Endpoint.ARROW)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)
            elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
                # Add bidirected edge j <-> i
                edge = Edge(nodes[j], nodes[i], Endpoint.ARROW, Endpoint.ARROW)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)

    return g_causal, edges

def maybe_plot_graph(graph, plot_name):
    """
    Plot a graph from a graph dictionary.
    Attributes:
        graph (Dataframe): A dictionary containing the graph information.
        plot_name (str): The name of the plot.

    Returns:
        None
    """
    if graph is not None:
        g = torch.Tensor(graph.values)
        labels = list(graph.index)
        g, edges_g = convert_adjMatrix_to_causallearnGraph(g, labels)
        # Convert graph to PyDot format
        pyd = GraphUtils.to_pydot(g, edges_g)
        
        # Create and read the PNG image of the graph
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        
        # Save the image as a PNG file without displaying it
        plt.imsave(f'{plot_name}.png', img)