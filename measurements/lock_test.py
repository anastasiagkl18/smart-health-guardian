import numpy as np, math, sys
d=np.load('raw1.npz'); t,ir=d['t'],d['ir']
fs=100.0
K=math.tan(math.pi*4/fs);Q=0.7071;n=1/(1+K/Q+K*K)
b0=K*K*n;b1=2*b0;b2=b0;a1=2*(K*K-1)*n;a2=(1-K/Q+K*K)*n
dc=ir[0];x1=x2=y1=y2=0.0;Y=np.zeros(len(ir))
for i,v in enumerate(ir):
    dc+=0.031*(v-dc); hp=dc-v
    yy=b0*hp+b1*x1+b2*x2-a1*y1-a2*y2; x2=x1;x1=hp;y2=y1;y1=yy; Y[i]=yy
S=np.diff(Y,prepend=Y[0])
def run(start, AMP_INIT=1.0, FALL_N=0, MEDIAN=False):
    Yv=Y[start:]; Sv=S[start:]; tv=t[start:]
    settle=150
    amp=AMP_INIT*(Yv[:settle].max()-Yv[:settle].min()); trough=Yv[settle]; state=0; last=-1e9; rr=0; miss=0
    beats=[]; hs=[]; fall=0
    for i in range(settle,len(Yv)):
        y=Yv[i]; s=Sv[i]; tm=tv[i]
        amp*=0.999
        if state==0:
            if y<trough: trough=y
            refr=max(250,min(0.6*rr,450))
            if y-trough>=0.5*amp and tm-last>=refr:
                state=1; peak=y; best=-1e9; bj=i; fall=0
                k=i
                while k>i-48 and Sv[k]>0:
                    if Sv[k]>best: best=Sv[k]; bj=k
                    k-=1
        else:
            if y>peak: peak=y
            if s>best and y>=peak: best=s; bj=i
            fall = fall+1 if s<0 else 0
            if y<peak-0.2*amp or (FALL_N and fall>=FALL_N):
                a_,b_,c_=Sv[bj-1],Sv[bj],Sv[bj+1]; den=a_-2*b_+c_
                off=0.5*(a_-c_)/den if den else 0; bt=tv[bj]-5+np.clip(off,-.5,.5)*10
                h=peak-trough
                if MEDIAN:
                    hs.append(h); hs=hs[-7:]; amp=float(np.median(hs)) if len(hs)>=3 else 0.8*amp+0.2*h
                else:
                    amp=0.8*amp+0.2*max(min(h,2*amp),0.5*amp)
                iv=bt-last
                if 300<iv<1500 and (rr==0 or 0.7*rr<iv<1.3*rr): rr=iv if rr==0 else 0.8*rr+0.2*iv; miss=0
                else:
                    miss+=1
                    if miss>=5: rr=0; miss=0
                beats.append(bt); last=bt; state=0; trough=y
    ibi=np.diff(beats); med=np.median(ibi[ibi<1000])
    lock=np.mean(ibi>1.6*med)*100; extra=np.mean(ibi<0.6*med)*100
    clean=ibi[(tv[0]+150000<np.array(beats[1:]))]
    ok=clean[(clean>450)&(clean<1000)]; sd=np.diff(ok)
    return len(beats), lock, extra, np.median(np.abs(sd))
starts=[0,1500,3000,6000,9000,12000,15000,20000]   # different finger-on moments
for cfg in [dict(),dict(AMP_INIT=0.5),dict(FALL_N=8),dict(FALL_N=8,AMP_INIT=0.5),dict(MEDIAN=True),dict(MEDIAN=True,FALL_N=8,AMP_INIT=0.5)]:
    res=[run(s,**cfg) for s in starts]
    print(f"{str(cfg):45s} missed(>1.6x)% {np.mean([r[1] for r in res]):5.1f}  worst {max(r[1] for r in res):5.1f}  extra% {np.mean([r[2] for r in res]):4.1f}  med|dIBI| clean {np.mean([r[3] for r in res]):5.1f}")
