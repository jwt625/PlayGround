

#%%
import plotly.graph_objects as go

# Define E-field components and dielectric tensor terms in Voigt order
e_fields = ['Eₓ', 'Eᵧ', 'E𝓏']
dielectric_terms_voigt = [
    'Δ(1/n₁²)',  # xx
    'Δ(1/n₂²)',  # yy
    'Δ(1/n₃²)',  # zz
    'Δ(1/n₂n₃)', # yz
    'Δ(1/n₁n₃)', # xz
    'Δ(1/n₁n₂)'  # xy
]
nodes = e_fields + dielectric_terms_voigt

# Mapping from label to index
label_to_index = {label: i for i, label in enumerate(nodes)}

# Electro-optic contributions based on r_ij (free boundary conditions)
# Format: (E-field, Dielectric term, Coefficient value)
links = [
    ('Eᵧ', 'Δ(1/n₁²)', 6.81),    # r22
    ('E𝓏', 'Δ(1/n₁²)', 10.0),    # r13
    ('Eᵧ', 'Δ(1/n₂²)', 6.81),    # r22
    ('E𝓏', 'Δ(1/n₂²)', 10.0),    # r13
    ('E𝓏', 'Δ(1/n₃²)', 32.2),    # r33
    ('Eᵧ', 'Δ(1/n₂n₃)', 32.0),   # r51
    ('Eₓ', 'Δ(1/n₁n₃)', 32.0),   # r51
    ('Eₓ', 'Δ(1/n₁n₂)', 6.81),   # r22
]

# Assign colors by E-field
field_colors = {
    'Eₓ': 'rgba(0, 100, 255, 0.5)',   # Blue
    'Eᵧ': 'rgba(255, 165, 0, 0.5)',   # Orange
    'E𝓏': 'rgba(200, 0, 0, 0.5)'      # Red
}

# Build source, target, value, color arrays
source = [label_to_index[s] for s, t, v in links]
target = [label_to_index[t] for s, t, v in links]
value  = [v for s, t, v in links]
color  = [field_colors[s] for s, t, v in links]

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=nodes,
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=color,
    ))])

# Save to HTML
fig.write_html("li_nb_eo_sankey_final.html")
print("Saved to li_nb_eo_sankey_final.html")

# %%
fig
# %%
from matplotlib.sankey import Sankey
Sankey(flows=[1.0, -0.5, -0.5], labels=['input', 'output1', 'output2']).finish()



# %%
