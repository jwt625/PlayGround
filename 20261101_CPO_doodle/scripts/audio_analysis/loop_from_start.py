"""Loop quality for each even bar count from the two preferred crop starts (12.931 s and 19.331 s).

Usage: uv run --no-project --with librosa --with numpy python scripts/audio_analysis/loop_from_start.py "Mittsies - Vitality.m4a"
Numbers only; nobody listened. Results are recorded in DevLog-000 section 12.
"""
import subprocess, sys, tempfile, os
import numpy as np, librosa

_tmp = os.path.join(tempfile.mkdtemp(), "track.wav")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", sys.argv[1], "-ac", "1", "-ar", "22050", _tmp], check=True)
y, sr = librosa.load(_tmp, sr=None, mono=True); dur = len(y) / sr; hop = 256
C=librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop); M=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr,hop_length=hop,n_mels=48))
F=np.vstack([(C-C.mean(1,keepdims=True))*4,(M-M.mean(1,keepdims=True))/20])
def corr(a,b,w):
    A=int(a*sr/hop);B=int(b*sr/hop);n=int(w*sr/hop)
    if B+n>F.shape[1] or A<0: return -1
    u=F[:,A:A+n].ravel();v=F[:,B:B+n].ravel()
    return float(np.dot(u-u.mean(),v-v.mean())/(np.linalg.norm(u-u.mean())*np.linalg.norm(v-v.mean())+1e-9))
rms=lambda a,b:20*np.log10(np.sqrt(np.mean(y[int(a*sr):int(b*sr)]**2))+1e-9)
print("bar-level energy 8.93 to 40:")
for t in np.arange(8.931,40,3.2): print(f"  {t:6.2f} {rms(t,t+3.2):6.1f} dB")
for st in (12.931,19.331):
    res=[]
    for bars in range(4,62,2):
        L=bars*1.6; en=st+L
        if en+3.2>dur: break
        after=corr(st,en,3.2); before=corr(st-3.2,en-3.2,3.2); lvl=abs(rms(st,st+.8)-rms(en-.8,en))
        res.append((after,bars,L,en,before,lvl))
    print(f"start {st}: top loop lengths (after-cut match, bars, s, end, before-cut match, level jump)")
    for r in sorted(res,reverse=True)[:8]: print(f"  {r[0]:.3f} {r[1]:3d} bars {r[2]:6.1f} s end {r[3]:7.3f}  before {r[4]:.3f}  {r[5]:.1f} dB")
    print("  even lengths under 34 bars:")
    for r in [q for q in res if q[1]<=34]: print(f"    {r[1]:3d} bars {r[2]:5.1f} s after {r[0]:.3f} before {r[4]:.3f} jump {r[5]:.1f}")
