import numpy as np, math
d=np.load('raw1.npz'); t,ir=d['t'],d['ir']
fs=100.0
K=math.tan(math.pi*4/fs);Q=0.7071;n=1/(1+K/Q+K*K)
b0=K*K*n;b1=2*b0;b2=b0;a1=2*(K*K-1)*n;a2=(1-K/Q+K*K)*n
# causal filtering exactly as on the device
dc=ir[0];x1=x2=y1=y2=0.0;Y=np.zeros(len(ir))
for i,v in enumerate(ir):
    dc+=0.031*(v-dc); hp=dc-v
    yy=b0*hp+b1*x1+b2*x2-a1*y1-a2*y2; x2=x1;x1=hp;y2=y1;y1=yy; Y[i]=yy
S=np.diff(Y,prepend=Y[0])
def run(fid, rise=0.5, drop=0.2, settle=150):
    amp=Y[:settle].max()-Y[:settle].min(); trough=Y[settle]; state=0; last=-1e9; rr=0; miss=0
    beats=[]; peak=0; i0=0
    for i in range(settle,len(Y)):
        y=Y[i]; s=S[i]; tm=t[i]
        amp*=0.999
        if state==0:
            if y<trough: trough=y; itr=i
            refr=max(250,min(0.6*rr,450))
            if y-trough>=rise*amp and tm-last>=refr:
                state=1; peak=y; ipk=i; i0=itr if 'itr' in dir() else i-15
        else:
            if y>peak: peak=y; ipk=i
            if y<peak-drop*amp:
                # fiducials between trough index i0 and peak index ipk
                seg=slice(max(i0,ipk-40),ipk+1)
                if fid=='slope':
                    k=max(i0,ipk-40)+int(np.argmax(S[seg]))
                    a_,b_,c_=S[k-1],S[k],S[k+1]; den=a_-2*b_+c_
                    off=0.5*(a_-c_)/den if den else 0; bt=t[k]-5+np.clip(off,-.5,.5)*10
                elif fid=='half':
                    lvl=trough+0.5*(peak-trough); k=ipk
                    while k>i0 and Y[k-1]>=lvl: k-=1
                    y0,y1_=Y[k-1],Y[k]; frac=(lvl-y0)/(y1_-y0) if y1_!=y0 else 0
                    bt=t[k-1]+frac*10
                elif fid=='peak':
                    k=ipk; a_,b_,c_=Y[k-1],Y[k],Y[k+1]; den=a_-2*b_+c_
                    off=0.5*(a_-c_)/den if den else 0; bt=t[k]+np.clip(off,-.5,.5)*10
                h=peak-trough; amp=0.8*amp+0.2*max(min(h,2*amp),0.5*amp)
                iv=bt-last
                if 300<iv<1500 and (rr==0 or 0.7*rr<iv<1.3*rr): rr=iv if rr==0 else 0.8*rr+0.2*iv; miss=0
                else:
                    miss+=1
                    if miss>=5: rr=0; miss=0
                beats.append(bt); last=bt; state=0; trough=y; itr=i
    return np.array(beats)
for fid in ['slope','half','peak']:
    bt=run(fid); ibi=np.diff(bt); tm=bt[1:]-t[0]
    for lo,hi in [(0,150e3),(150e3,260e3)]:
        x=ibi[(tm>=lo)&(tm<hi)]; ok=x[(x>450)&(x<1000)]; sd=np.diff(ok)
        # 18% filter like firmware
        f=[(a,b) for a,b in zip(ok,ok[1:]) if abs(b-a)<0.18*a]; rf=math.sqrt(np.mean([(b-a)**2 for a,b in f]))
        print(f'{fid:5s} {lo/1000:3.0f}-{hi/1000:3.0f}s beats {len(x):3d} HR {60000/np.median(x):5.1f}  outliers {100*np.mean((x<450)|(x>1000)):4.1f}%  med|dIBI| {np.median(np.abs(sd)):5.1f}  RMSSD {np.sqrt(np.mean(sd**2)):6.1f}  RMSSD(18%) {rf:5.1f}')
