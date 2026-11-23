"""Normal-incidence reflectance colour of thin-film stacks on Si (transfer matrix, CIE 1931, D65-like
blackbody 6504 K, linear sRGB). Plain Python 3 + numpy; prints the colours used for the LNOI and SiN
chip surfaces in tfln_die.py, shot_pyro_damage.py and shot_sin_ring.py. Si n,k are coarse approximate
values typed from memory (not a cited table); LN index 2.22 (no birefringence)."""
import numpy as np
lam=np.linspace(380,720,341)
def g(x,mu,s1,s2): return np.exp(-0.5*((x-mu)/np.where(x<mu,s1,s2))**2)
xb=1.056*g(lam,599.8,37.9,31.0)+0.362*g(lam,442.0,16.0,26.7)-0.065*g(lam,501.1,20.4,26.2)
yb=0.821*g(lam,568.8,46.9,40.5)+0.286*g(lam,530.9,16.3,31.1)
zb=1.217*g(lam,437.0,11.8,36.0)+0.681*g(lam,459.0,26.0,13.8)
# D65 approx via blackbody 6504K
h,c,k=6.626e-34,3e8,1.381e-23
L=lam*1e-9; S=1/L**5/(np.exp(h*c/(L*k*6504))-1); S/=S.max()
def nSi(l):  # rough Si dispersion (approximate values)
    pts=np.array([[400,5.57,0.39],[450,4.67,0.13],[500,4.29,0.07],[550,4.08,0.04],[600,3.94,0.025],[650,3.85,0.016],[700,3.78,0.01],[720,3.76,0.009]])
    return np.interp(l,pts[:,0],pts[:,1])+1j*np.interp(l,pts[:,0],pts[:,2])
def R(layers):
    out=[]
    for l in lam:
        ns=[1.0]+[n for n,d in layers]+[nSi(l)]
        ds=[d for n,d in layers]
        r=(ns[-2]-ns[-1])/(ns[-2]+ns[-1]); 
        # build from substrate up
        rt=r
        for i in range(len(layers)-1,-1,-1):
            n0=ns[i]; n1=ns[i+1]; d=ds[i]
            r01=(n0-n1)/(n0+n1); beta=2*np.pi*n1*d/l
            rt=(r01+rt*np.exp(2j*beta))/(1+r01*rt*np.exp(2j*beta))
        out.append(abs(rt)**2)
    return np.array(out)
def rgb(Rs):
    X=np.sum(S*xb*Rs);Y=np.sum(S*yb*Rs);Z=np.sum(S*zb*Rs)
    Y0=np.sum(S*yb); X,Y,Z=X/Y0,Y/Y0,Z/Y0
    M=np.array([[3.2406,-1.5372,-0.4986],[-0.9689,1.8758,0.0415],[0.0557,-0.2040,1.0570]])
    # white normalize
    Xw,Yw,Zw=np.sum(S*xb)/Y0,1,np.sum(S*zb)/Y0
    w=M@[Xw,Yw,Zw]; v=M@[X,Y,Z]
    return np.clip(v/w,0,None)
for name,lay in [("bareSi",[]),("field LN300/ox4.7",[(2.22,300),(1.46,4700)]),("rib LN600/ox4.7",[(2.22,600),(1.46,4700)]),
 ("field LN300/ox2",[(2.22,300),(1.46,2000)]),("rib LN600/ox2",[(2.22,600),(1.46,2000)]),
 ("cladSiO2 1um + LN300/ox4.7",[(1.46,1000),(2.22,300),(1.46,4700)]),
 ("SiN ring: clad2.5/SiN0.8/ox4", [(1.46,2500),(2.0,800),(1.46,4000)]),("SiN field: ox 6.5+0.8",[(1.46,7300)]),
 ("field LN 600 unetched/ox4.7",[(2.22,600),(1.46,4700)]), ("bulk LN-like?",[(2.22,500000)])]:
    if name.startswith("bulk"): continue
    r=R(lay); print(f"{name:30s} rgb_lin={np.round(rgb(r),3)}  meanR={r.mean():.3f}")


# ---------------------------------------------------------------------------------------------
# Microscope (partially coherent) model for thick-oxide chips: layers thicker than the
# illumination coherence length (here the um-scale SiO2 cladding and BOX) are added
# incoherently (intensities); only the thin SiN core film interferes coherently.
def _r_film(l, n_out, n_f, d):
    r01 = (n_out - n_f) / (n_out + n_f)
    b = 2 * np.pi * n_f * d / l
    r = (r01 - r01 * np.exp(2j * b)) / (1 - r01 * r01 * np.exp(2j * b))
    return abs(r) ** 2


def incoherent_sin(d_sin):
    out = []
    for l in lam:
        ra = ((1 - 1.46) / (1 + 1.46)) ** 2
        rb = _r_film(l, 1.46, 2.0, d_sin) if d_sin > 0 else 0.0
        ns = nSi(l)
        rc = abs((1.46 - ns) / (1.46 + ns)) ** 2
        rbc = rb + (1 - rb) ** 2 * rc / (1 - rb * rc)
        out.append(ra + (1 - ra) ** 2 * rbc / (1 - ra * rbc))
    return np.array(out)


if __name__ == "__main__":
    print("partially coherent microscope model (thick SiO2 incoherent):")
    for d in (0, 300, 400, 800):
        c = rgb(incoherent_sin(d))
        print(f"  SiN {d:4d} nm  rgb_lin={np.round(c, 3)}  Y={0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]:.3f}")
