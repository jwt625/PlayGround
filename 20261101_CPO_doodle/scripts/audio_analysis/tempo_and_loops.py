"""Whole-track tempo grid, phase stability, sections, near-loop crops on bar lines, rhythm subdivisions.

Usage: uv run --no-project --with librosa --with numpy python scripts/audio_analysis/tempo_and_loops.py "Mittsies - Vitality.m4a"
Numbers only; nobody listened. Results are recorded in DevLog-000 section 12.
"""
import subprocess, sys, tempfile, os
import numpy as np, librosa

_tmp = os.path.join(tempfile.mkdtemp(), "track.wav")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", sys.argv[1], "-ac", "1", "-ar", "22050", _tmp], check=True)
y, sr = librosa.load(_tmp, sr=None, mono=True); dur = len(y) / sr; hop = 256
oe=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
def score(per,lo=0,hi=dur):
    best=(-1,0); a=int(lo*sr/hop); b=int(hi*sr/hop); o=oe[a:b]; m=o.mean()
    for ph in np.linspace(0,per,64,endpoint=False):
        idx=(np.arange(ph,(hi-lo)-0.01,per)*sr/hop).astype(int); idx=idx[idx<len(o)]
        s=o[idx].mean()/m
        if s>best[0]: best=(s,ph)
    return best
print("whole-track grid contrast by BPM:")
for bpm in (75,99.4,100,148,149,149.5,150,150.5,151,152,153,155,160,174,175):
    s,ph=score(60/bpm); print(f"  {bpm:6.1f}: {s:.2f}")
fine=[(score(60/b)[0],b) for b in np.arange(149.5,150.51,0.02)]; c,b=max(fine); per=60/b; print(f"best fine {b:.2f} BPM contrast {c:.2f}")
per=0.4
print("phase at 150 BPM per 30 s window (stable phase = constant tempo):")
ph0=score(per)[1]
for t0 in np.arange(0,dur-30,30):
    s,ph=score(per,t0,t0+30); abs_ph=(t0+ph)%per
    print(f"  {t0:5.0f}: contrast {s:.2f} phase {abs_ph:.3f}")
print("global phase",round(ph0,3))
# Bars: bar phase from recurring harmonic change, candidates over 4 offsets
C=librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop); M=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr,hop_length=hop,n_mels=48))
beats=np.arange(ph0,dur-per,per); bi=(beats*sr/hop).astype(int)
Cb=np.array([C[:,bi[i]:bi[i+1]].mean(1) for i in range(len(bi)-1)]).T
chg=np.r_[0,np.linalg.norm(np.diff(Cb,axis=1),axis=0)]
print("chroma change by beat-of-4:",[f"{chg[j::4].mean():.3f}" for j in range(4)])
off=int(np.argmax([chg[j::4].mean() for j in range(4)])); bar0=ph0+off*per; print(f"bar phase: first downbeat {bar0:.3f} s")
# Energy and sections per 8 bars
print("sections every 8 bars (12.8 s), start, rms dB, flatness of onset, similarity to next 8 bars:")
Mb=M-M.mean(1,keepdims=True); Fb=np.vstack([(C-C.mean(1,keepdims=True))*4,Mb/20])
def corr(a,b,w):
    A=int(a*sr/hop);B=int(b*sr/hop);n=int(w*sr/hop)
    if B+n>Fb.shape[1] or A<0: return -1
    u=Fb[:,A:A+n].ravel();v=Fb[:,B:B+n].ravel()
    return float(np.dot(u-u.mean(),v-v.mean())/(np.linalg.norm(u-u.mean())*np.linalg.norm(v-v.mean())+1e-9))
for t in np.arange(bar0,dur-12.8,12.8):
    a=int(t*sr);b=int((t+12.8)*sr);r=20*np.log10(np.sqrt(np.mean(y[a:b]**2))+1e-9)
    print(f"  {t:7.2f} rms {r:6.1f}  next8 {corr(t,t+12.8,12.8):.2f}  next4 {corr(t,t+6.4,6.4):.2f}")
# Near-loop on bar lines, with mean removed (time pattern only)
res=[]
for bars in (4,6,8,10,12,16,20,24):
    L=bars*1.6
    for st in np.arange(bar0%1.6,dur-L-3.2,1.6):
        en=st+L
        if st<3.2: continue
        after=corr(st,en,3.2); before=corr(st-3.2,en-3.2,3.2)
        ra=np.sqrt(np.mean(y[int(st*sr):int((st+.4)*sr)]**2)); rb=np.sqrt(np.mean(y[int((en-.4)*sr):int(en*sr)]**2)); lvl=abs(20*np.log10((ra+1e-9)/(rb+1e-9)))
        res.append((min(after,before),bars,L,st,en,after,before,lvl))
res.sort(reverse=True)
print("near-loop on bar lines, mean removed (min score, bars, s, start, end, after, before, level jump):")
for bars in (8,12,16):
    k=0
    for r in res:
        if r[1]==bars:
            print(f"  {r[0]:.3f} {r[1]:3d} bars {r[2]:6.2f} s {r[3]:7.3f} -> {r[4]:7.3f}  {r[5]:.3f} {r[6]:.3f} {r[7]:.1f} dB"); k+=1
            if k==5: break
# Rhythm inside the best crops: autocorrelation of onset at musical lags
lags=[0.1,0.2,0.3,0.4,0.6,0.8,1.2,1.6]
for name,a,b in [("best 12 bars",)+tuple(next(r for r in res if r[1]==12)[3:5]),("best 8 bars",)+tuple(next(r for r in res if r[1]==8)[3:5])]:
    i=int(a*sr/hop); j=int(b*sr/hop); o=oe[i:j]-oe[i:j].mean(); ac=np.correlate(o,o,'full')[len(o)-1:]; ac/=ac[0]
    print(name,f"{a:.3f}-{b:.3f} onset autocorr:",{l:round(float(ac[int(l*sr/hop)]),2) for l in lags})
