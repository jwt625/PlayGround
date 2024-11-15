
#%%
import gdsfactory as gf

# %%
c = gf.Component("test")

# Create some new geometry from the functions available in the geometry library
t = gf.components.text("DINNER")
c << t

ref_rect = c << gf.components.rectangle(size=(2,2), layer=(1,0),centered=True)
ref_rect.move((2,5))

ref_rect = c << gf.components.rectangle(size=(2,2), layer=(1,0),centered=True)
ref_rect.move((39,5))

ref_rect = c << gf.components.rectangle(size=(2,3), layer=(1,0),centered=True)
ref_rect.move((44, 4.5))

c.write_gds('tmp.gds')

# %%
