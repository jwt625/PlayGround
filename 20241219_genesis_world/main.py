

#%% test

import genesis as gs
gs.init(backend=gs.cpu)

# #%%
# from genesis.ext import pyrender, trimesh

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build()

for i in range(1000):
    scene.step()
# %%