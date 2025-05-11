

#%%
import plotly.graph_objects as go

# Create an interactive neural network-like diagram using Plotly
# This is just a small 3-layer NN: input → hidden → output

# Define layer node positions
layer_positions = {
    'input':  [(0, 2), (0, 1), (0, 0)],
    'hidden': [(1, 2.5), (1, 1.5), (1, 0.5), (1, -0.5)],
    'output': [(2, 1.5), (2, 0.5)]
}

# Assign node labels
input_nodes = ['x₁', 'x₂', 'x₃']
hidden_nodes = ['h₁', 'h₂', 'h₃', 'h₄']
output_nodes = ['y₁', 'y₂']
all_nodes = input_nodes + hidden_nodes + output_nodes
positions = layer_positions['input'] + layer_positions['hidden'] + layer_positions['output']

# Create node trace
node_trace = go.Scatter(
    x=[p[0] for p in positions],
    y=[p[1] for p in positions],
    text=all_nodes,
    mode='markers+text',
    textposition='middle right',
    marker=dict(size=20, color='lightblue', line=dict(width=2, color='black'))
)

# Create edge traces (just connect each layer fully for demo)
edge_x, edge_y = [], []
for i_idx, i_pos in enumerate(layer_positions['input']):
    for h_idx, h_pos in enumerate(layer_positions['hidden']):
        edge_x += [i_pos[0], h_pos[0], None]
        edge_y += [i_pos[1], h_pos[1], None]
for h_idx, h_pos in enumerate(layer_positions['hidden']):
    for o_idx, o_pos in enumerate(layer_positions['output']):
        edge_x += [h_pos[0], o_pos[0], None]
        edge_y += [h_pos[1], o_pos[1], None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=2, color='gray'),
    hoverinfo='none',
    mode='lines'
)

# Combine into figure
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title='Interactive Neural Network Architecture',
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    height=500
)

fig.show()














































# %%
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# Define weights (3 input → 4 hidden → 2 output)
# Define weights correctly for 3 input → 4 hidden → 2 output
W1 = np.array([
    [0.5, 1.2, -1.0, 0.0],   # x1 → h1–h4
    [-0.4, 0.8, 0.3, 0.6],   # x2 → h1–h4
    [0.1, -0.6, 0.5, 1.0]    # x3 → h1–h4
])  # shape: (3 input, 4 hidden)



W2 = np.array([
    [1.0, -0.6, 0.8, 0.2],
    [-0.3, 0.9, -1.0, 0.5]
])  # shape: (2 output, 4 hidden)

def relu(x): return np.maximum(0, x)

# Layer positions
layer_x = [0, 1, 2]
input_y = [2, 1, 0]
hidden_y = [2.5, 1.5, 0.5, -0.5]
output_y = [1.5, 0.5]

node_x = [layer_x[0]] * 3 + [layer_x[1]] * 4 + [layer_x[2]] * 2
node_y = input_y + hidden_y + output_y
node_labels = ['x₁', 'x₂', 'x₃', 'h₁', 'h₂', 'h₃', 'h₄', 'y₁', 'y₂']

def get_node_colors(activations):
    return ['lightgray'] * 3 + [f'rgba(0,0,255,{min(1,a)})' for a in activations['hidden']] + \
           [f'rgba(255,0,0,{min(1,a)})' for a in activations['output']]

def get_edge_traces(x_vals):
    x = np.array(x_vals)
    h_raw = x @ W1
    h = relu(h_raw)
    y = W2 @ h

    act = {
        'hidden': h / (h.max() if h.max() != 0 else 1),
        'output': y / (np.abs(y).max() if np.abs(y).max() != 0 else 1)
    }
    node_colors = get_node_colors(act)

    edge_x, edge_y, edge_colors = [], [], []

    # Input → Hidden
    for i in range(3):
        for j in range(4):
            edge_x += [layer_x[0], layer_x[1], None]
            edge_y += [input_y[i], hidden_y[j], None]
            val = x[i] * W1[i, j]
            opacity = min(1, abs(val) / 2)
            color = f'rgba(0,0,0,{opacity})' if val != 0 else 'rgba(200,200,200,0.1)'
            edge_colors.append(color)

    # Hidden → Output
    for i in range(4):
        for j in range(2):
            edge_x += [layer_x[1], layer_x[2], None]
            edge_y += [hidden_y[i], output_y[j], None]
            val = h[i] * W2[j, i]
            opacity = min(1, abs(val) / 2)
            color = f'rgba(0,0,0,{opacity})' if val != 0 else 'rgba(200,200,200,0.1)'
            edge_colors.append(color)

    return edge_x, edge_y, edge_colors, node_colors

# Plotly FigureWidget
fig = go.FigureWidget()
edge_trace = go.Scatter(x=[], y=[], mode='lines', line=dict(width=3), hoverinfo='none')
node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                        marker=dict(size=30, color=[]),
                        text=node_labels, textposition='middle right')
fig.add_trace(edge_trace)
fig.add_trace(node_trace)
fig.update_layout(title='Neural Network Activation Flow',
                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                  showlegend=False, plot_bgcolor='white', width=700, height=500)

def update_plot(x1=1.0, x2=0.5, x3=0.0):
    edge_x, edge_y, edge_colors, node_colors = get_edge_traces([x1, x2, x3])
    fig.data[0].x = edge_x
    fig.data[0].y = edge_y
    fig.data[0].line.color = edge_colors
    fig.data[1].marker.color = node_colors

# Interactive sliders
x1_slider = widgets.FloatSlider(value=1.0, min=-2, max=2, step=0.1, description='x₁')
x2_slider = widgets.FloatSlider(value=0.5, min=-2, max=2, step=0.1, description='x₂')
x3_slider = widgets.FloatSlider(value=0.0, min=-2, max=2, step=0.1, description='x₃')

widgets.interact(update_plot, x1=x1_slider, x2=x2_slider, x3=x3_slider)
display(fig)

# %%
