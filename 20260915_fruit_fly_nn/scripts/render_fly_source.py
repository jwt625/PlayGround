"""Save source MuJoCo visual-group render for comparison with GLB preview."""
from pathlib import Path
import mujoco
from PIL import Image
root=Path(__file__).resolve().parents[1]
m=mujoco.MjModel.from_xml_path(str(root/'assets/vendor/flybody/assets/fruitfly.xml'))
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,height=480,width=640)
opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
cam=mujoco.MjvCamera();cam.lookat[:]=[-.04,0,-.02];cam.distance=1.;cam.azimuth=90;cam.elevation=0
r.update_scene(d,camera=cam,scene_option=opt)
p=root/'assets/generated/flybody/validation/mujoco-side.png';p.parent.mkdir(parents=True,exist_ok=True)
Image.fromarray(r.render()).save(p);r.close();print(p)
