"""Exact local mean derivative, cheap linear transport of an extra response."""
import flopscope as fs
import flopscope.numpy as f

def mean_gain(mu,var,d3,g4):
    s=f.sqrt(var);a=mu/s
    a2=a*a;a3=a2*a
    h2=a2-1.;h3=a3-3.*a
    h5=a3*a2-10.*a3+15.*a
    h6=a2*a2*a2-15.*a2*a2+45.*a2-15.
    k3=d3/(var*s);k4=g4/(var*var)
    polynomial=k3*h2/6.-k4*h3/24.-k3*k3*h5/72.+k3*k4*h6/144.
    return fs.stats.norm.cdf(a)+fs.stats.norm.pdf(a)*polynomial

def response(cap,weights,forcing):
    delta=f.zeros(weights.shape[1],dtype=f.float64);rows=[]
    for li in range(weights.shape[0]):
        if li<8:rows.append(f.copy(delta));continue
        gain=mean_gain(*(cap[f'{key}_{li}'] for key in ('mu','var','d3','g4')))
        if li>8:delta=gain*(weights[li].T@delta)
        if li in forcing:delta=delta+forcing[li]
        rows.append(delta)
    return f.maximum(cap['prediction']+f.stack(rows),0.)
