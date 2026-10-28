# MIT License
#
# Copyright (c) 2026 504aldo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Frozen quartic response with coefficient contraction before feature assembly.

Same source polynomials and shared coefficients as the captured response.
All numerical work uses public FlopScope; no reference or network ID is read.
"""
from pathlib import Path
import flopscope.numpy as f
from _zero_store import zero32 as _zero32

def birth_meta(mu,a,s,phi,g,cov,sketch,wick,ids,coefficients,ones,H_buf=None):
    # Gaussian raw moments and derivatives, already calculated in the parent.
    # Keep this separate from the Edgeworth-corrected postactivation moments.
    v=s*s;h2=wick[:,6];h3=wick[:,8]
    m=wick[:,0]
    raw=f.concatenate((ones[None],wick[:,:4].T),axis=0)
    terms=f.prod(f.take(raw,ids,axis=0),axis=1)
    k4=f.einsum('i,in->n',coefficients,terms,optimize=False)
    c22=12*(g-g*g-m*h2);leaf=g*v
    n=s.shape[0];rank=min(8,n);qw=min(n,rank+4)
    h2_c=h2[:,None]
    d_h=(v*(h2*h2))[:,None]
    q,_=f.linalg.qr(sketch[:,:qw])
    for _ in range(2):
        Hq=h2_c*(cov@(h2_c*q))+d_h*q
        q,_=f.linalg.qr(Hq)
    Hq=h2_c*(cov@(h2_c*q))+d_h*q
    small=q.T@Hq
    values,rotation=f.linalg.eigh((small+small.T)*0.5)
    chosen=f.argsort(f.abs(values))[-rank:]
    values=f.take(values,chosen);vectors=q@f.take(rotation,chosen,axis=1)
    diagonal=f.einsum('ir,ir,r->i',vectors,vectors,values,optimize=False)
    h34=4*h3
    counter=k4-f.einsum('i,i,i->i',leaf,leaf,leaf*h34+c22+12*diagonal,optimize=False)
    return f.stack((leaf,h34,c22,counter),axis=0),vectors,values

def source_marginals(A,P,metadata,bufs=None,bcs=None,diag_out=None,traces_out=None,proj_out=None):
    co,vectors,values=metadata
    if bufs is None or diag_out is None or traces_out is None or proj_out is None:
        B=A+P*co[:,0,None,:]
        pp=P*P;bb=B*B;pb=P*B
        star=f.einsum('kij,kij,kj->ki',pb,bb,co[:,1],optimize=False)
        repeat=f.einsum('kij,kij,kj->ki',pp,bb,co[:,2],optimize=False)
        counter=f.einsum('kij,kij,kj->ki',pp,pp,co[:,3],optimize=False)
        projected=pb@vectors
        path=12*f.einsum('kiq,kiq,kq->ki',projected,projected,values,optimize=False)
        pn=f.sum(pp,axis=1);bn=f.sum(bb,axis=1);dot=f.sum(pb,axis=1)
        t0=f.einsum('kj,kj,kj->k',co[:,1],dot,bn,optimize=False)
        t1=(f.einsum('kj,kj,kj->k',co[:,2],pn,bn,optimize=False)+2*f.einsum('kj,kj,kj->k',co[:,2],dot,dot,optimize=False))/3.0
        t2=f.einsum('kj,kj,kj->k',co[:,3],pn,pn,optimize=False)
        return f.stack((star,repeat,path,counter),axis=1),f.stack((t0,t1,t2),axis=1)
    k,ns,n_full=A.shape
    if bcs is not None and len(bcs)>=k:
        groups=[]
        lo=0
        while lo<k:
            end=lo+1;bc=bcs[lo]
            while end<k and bcs[end]==bc:end+=1
            groups.append((lo,end,bc));lo=end
    else:
        groups=[(0,k,n_full)]
    for lo,hi,bc in groups:
        A_m=A[lo:hi,:ns,:bc];P_m=P[lo:hi,:ns,:bc];co_m=co[lo:hi,:,:bc]
        B_m=f.multiply(P_m,co_m[:,0,None,:],out=bufs['mp'][lo:hi,:ns,:bc])
        f.add(A_m,B_m,out=B_m)
        pp_m=bufs['pp'][lo:hi,:ns,:bc]
        bb_m=f.multiply(B_m,B_m,out=bufs['t'][lo:hi,:ns,:bc])
        pb_m=f.multiply(P_m,B_m,out=bufs['ap'][lo:hi,:ns,:bc])
        f.einsum('kij,kij,kj->ki',pb_m,bb_m,co_m[:,1],optimize=False,out=diag_out[lo:hi,0,:ns])
        f.einsum('kij,kij,kj->ki',pp_m,bb_m,co_m[:,2],optimize=False,out=diag_out[lo:hi,1,:ns])
        f.einsum('kij,kij,kj->ki',pp_m,pp_m,co_m[:,3],optimize=False,out=diag_out[lo:hi,3,:ns])
        pn_m=f.sum(pp_m,axis=1);bn_m=f.sum(bb_m,axis=1);dot_m=f.sum(pb_m,axis=1)
        f.einsum('kj,kj,kj->k',co_m[:,1],dot_m,bn_m,optimize=False,out=traces_out[lo:hi,0])
        f.einsum('kj,kj,kj->k',co_m[:,2],pn_m,bn_m,optimize=False,out=traces_out[lo:hi,1])
        f.add(traces_out[lo:hi,1],2*f.einsum('kj,kj,kj->k',co_m[:,2],dot_m,dot_m,optimize=False),out=traces_out[lo:hi,1])
        f.divide(traces_out[lo:hi,1],3.0,out=traces_out[lo:hi,1])
        f.einsum('kj,kj,kj->k',co_m[:,3],pn_m,pn_m,optimize=False,out=traces_out[lo:hi,2])
        f.matmul(pb_m,vectors[lo:hi,:bc,:],out=proj_out[lo:hi,:ns,:])
    f.einsum('kiq,kiq,kq->ki',proj_out[:k,:ns,:],proj_out[:k,:ns,:],values,optimize=False,out=diag_out[:k,2,:ns])
    f.multiply(diag_out[:k,2,:ns],12.0,out=diag_out[:k,2,:ns])
    if ns<diag_out.shape[2]:
        f.copyto(diag_out[:k,:,ns:],_zero32())
    return diag_out[:k],traces_out[:k]

def local_response(a,s,d3,k4,theta,powers):
    a=f.clip(a,-6.0,6.0)
    phi=f.exp(a*a*(-0.5))*0.3989422804014327
    v=s*s
    k3=f.clip(d3/(v*s),-4.0,4.0)
    k4=f.clip(k4/(v*v),-4.0,4.0)
    H=f.vander(a,6,increasing=True).astype(f.float32)
    K=f.stack((f.ones_like(a),k3,k4),axis=1)
    quadratic=(H@theta).reshape(a.shape[0],3,3)
    return (s*phi)*f.einsum('ni,nij,nj->n',K,quadratic,K,optimize=False)

def prepare_weights(self,depth):
    # Finite architecture metadata, prepared before prediction when possible.
    hermite_to_power=f.asarray([[1,0,-1,0,3,0],[0,1,0,-3,0,15],[0,0,1,0,-6,0],
                                [0,0,0,1,0,-10],[0,0,0,0,1,0],[0,0,0,0,0,1]],dtype=f.float32)
    self._higher_powers=f.asarray([0,1,2,3,4,5],dtype=f.float32)
    local=hermite_to_power[None]@self.HIGHER_COEFFICIENTS[:,:36].reshape(4,6,6)
    # Store each quadratic monomial once, with no extra 1/2 factors.
    expansion=f.asarray([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],
      [0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],
      [0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1]],dtype=f.float32)
    local=local@expansion
    self._higher_moment_ids=f.asarray([[1,1,1,1],[1,1,2,0],[1,3,0,0],[2,2,0,0],[4,0,0,0]],dtype=f.int32)
    self._higher_moment_coefficients=f.asarray([-6,12,-4,-3,1],dtype=f.float32)
    weights=[]
    for li in range(depth):
        group=min(3,(li*4)//depth)
        ages=[li-j for j in range(li)]
        bins=[0 if age==1 else 1 if age<4 else 2 if age<8 else 3 for age in ages]
        ids=f.asarray(bins,dtype=f.int32)
        row=self.HIGHER_COEFFICIENTS[group]
        weights.append((local[group],f.take(row[36:].reshape(4,9),ids,axis=0)))
    self._higher_weights=weights

def _ensure_higher_buffers(self, n, depth):
    cache = self.__dict__.setdefault('_higher_buf_cache', {})
    key = (n, depth)
    entry = cache.get(key)
    if entry is None:
        ones = f.ones(n, dtype=f.float32)
        h_buf = None
        s0 = f.empty((max(0, depth - 1), 4, n), dtype=f.float32)
        f.copyto(s0, _zero32())
        s1 = f.empty((max(0, depth - 1), n, min(8, n)), dtype=f.float32)
        f.copyto(s1, _zero32())
        s2 = f.empty((max(0, depth - 1), min(8, n)), dtype=f.float32)
        f.copyto(s2, _zero32())
        tot_pairs = max(1, depth * (depth - 1) // 2)
        diag_all = f.empty((tot_pairs, 4, n), dtype=f.float32)
        f.copyto(diag_all, _zero32())
        traces_all = f.empty((tot_pairs, 3), dtype=f.float32)
        f.copyto(traces_all, _zero32())
        proj_buf = f.empty((max(1, depth - 1), n, min(8, n)), dtype=f.float32)
        f.copyto(proj_buf, _zero32())
        entry = (ones, h_buf, (s0, s1, s2), diag_all, traces_all, proj_buf)
        cache[key] = entry
    self._higher_ones, self._higher_H_buf, self._higher_storage, self._higher_diag_all, self._higher_traces_all, self._higher_proj_buf = entry
    wcache = self.__dict__.setdefault('_higher_w_cache', {})
    wentry = wcache.get(depth)
    if wentry is None:
        prepare_weights(self, depth)
        prepare_deferred(self, depth)
        wentry = (
            self._higher_weights,
            getattr(self, '_higher_diag_weights', None),
            getattr(self, '_higher_trace_weights', None),
            getattr(self, '_higher_local_weights', None),
        )
        wcache[depth] = wentry
    else:
        (
            self._higher_weights,
            self._higher_diag_weights,
            self._higher_trace_weights,
            self._higher_local_weights,
        ) = wentry

def higher_observe(self,W,mu,a,s,phi,g,d3,k4,cov,sketch,li,depth,inverse,wick):
    n=a.shape[0]
    if li==0:
        self._higher_meta=[];self._higher_last=None
        self._higher_records=[];self._higher_diags=[];self._higher_traces=[]
        self._higher_inverses=[];self._higher_covnorm=[]
        _ensure_higher_buffers(self, n, depth)
    else:
        if getattr(self,'_higher_diag_all',None) is None:
            diag,traces=self._higher_last
            if diag.shape[2]<n:diag=f.pad(diag,((0,0),(0,0),(0,n-diag.shape[2])),mode='constant')
            self._higher_diags.append(diag);self._higher_traces.append(traces)
        # Stack copies any ephemeral views before the parent reuses its pool.
        self._higher_records.append(f.stack((a,s,phi,g,d3,k4),axis=0))
        self._higher_inverses.append(f.copy(inverse))
        _csq_m7 = getattr(self, '_cur_c_sq', None)
        self._higher_covnorm.append(f.asarray(0.,dtype=f.float32) if cov is None else (f.sum(_csq_m7) if _csq_m7 is not None else f.einsum('ij,ij->',cov,cov,optimize=False)))
        self._higher_last=None
    if li+1<depth:
        metadata=birth_meta(mu,a,s,phi,g,cov,sketch,wick,self._higher_moment_ids,self._higher_moment_coefficients,self._higher_ones,self._higher_H_buf)
        for target,value in zip(self._higher_storage,metadata):f.copyto(target[li],value)
        self._higher_meta.append(None)

def prepare_deferred(self,depth):
    if depth<2:return
    total=depth*(depth-1)//2
    wd=f.zeros((depth-1,total,4),dtype=f.float32)
    wt=f.zeros((depth-1,total,3),dtype=f.float32)
    lo=0;local=[]
    for li in range(1,depth):
        theta,weights=self._higher_weights[li]
        local.append(theta)
        f.copyto(wd[li-1,lo:lo+li],weights[:,:4])
        f.copyto(wt[li-1,lo:lo+li],weights[:,6:9])
        lo+=li
    self._higher_diag_weights=wd;self._higher_trace_weights=wt
    self._higher_local_weights=f.stack(local,axis=0)

def finish_response(self,base,mlp):
    if len(self._higher_records)==0:return base
    data=f.stack(self._higher_records,axis=0)
    a,s,phi,g,d3,k4=(data[:,j] for j in range(6))
    depth,n=base.shape
    aa=f.clip(a,-6.0,6.0)
    density=f.exp(aa*aa*(-0.5))*0.3989422804014327
    v=s*s
    k3=f.clip(d3/(v*s),-4.0,4.0);kk4=f.clip(k4/(v*v),-4.0,4.0)
    H=f.vander(aa.reshape(-1),6,increasing=True).astype(f.float32).reshape(depth-1,n,6)
    K=f.stack((f.ones_like(aa),k3,kk4),axis=2)
    quadratic=f.einsum('lni,lij->lnj',H,self._higher_local_weights,optimize=False).reshape(depth-1,n,3,3)
    fresh=(s*density)*f.einsum('lni,lnij,lnj->ln',K,quadratic,K,optimize=False)
    _tot_m07=depth*(depth-1)//2
    if getattr(self,'_higher_diag_all',None) is not None and not self._higher_diags:
        diag=self._higher_diag_all[:_tot_m07]
        traces=self._higher_traces_all[:_tot_m07]
    else:
        diag=f.concatenate(self._higher_diags,axis=0)
        traces=f.concatenate(self._higher_traces,axis=0)
    value=f.einsum('kaj,lka->lj',diag,self._higher_diag_weights,optimize=False)
    trace_value=f.einsum('ka,lka->l',traces,self._higher_trace_weights,optimize=False)
    h4=(a*a-1)*phi/(v*s)/24
    sumv=f.sum(v,axis=1)
    den=sumv*sumv+2*(f.sum(v*v,axis=1)+f.stack(self._higher_covnorm))
    cup=3*h4*v*v/f.maximum(den[:,None],1e-20)
    fresh=fresh+h4*value+cup*trace_value[:,None]
    inverse=getattr(self,'_higher_inverse_stack',None)
    if inverse is None:inverse=f.stack(self._higher_inverses,axis=0)
    self._higher_inverse_stack=None
    fresh=f.take_along_axis(fresh,inverse,axis=1)
    scale=getattr(self,'_higher_fresh_scale',None)
    if scale is not None and scale.shape[0]==fresh.shape[0]:fresh=fresh*scale[:,None]
    gains=f.take_along_axis(g,inverse,axis=1)
    out=f.empty(base.shape,dtype=f.float32);f.copyto(out[0],base[0])
    wt=self._readout_weight_transposes
    error=fresh[0]
    errors=[error]
    we_list=[]
    for li in range(2,depth):
        we=wt[li]@error
        we_list.append(we)
        error=gains[li-1]*we+fresh[li-1]
        errors.append(error)
    errors_st=f.stack(errors,axis=0)
    self._m7_we_1_14=f.stack(we_list,axis=0) if we_list else None
    self._m7_errors=errors_st
    f.add(base[1:],errors_st,out=out[1:])
    self._higher_records=[];self._higher_diags=[];self._higher_traces=[]
    self._higher_inverses=[];self._higher_covnorm=[]
    return out

def install(cls):
    original=cls._dslices;setup0=cls.setup
    def wrapped(self,A,P,Z,L,w2,s,e,c1,c2,y,n,bufs,*args,**kwargs):
        result=original(self,A,P,Z,L,w2,s,e,c1,c2,y,n,bufs,*args,**kwargs)
        if getattr(self,'MARGINAL_MODE','off')!='off':
            ns=A.shape[1] if self._output_ids is None else self._output_ids.shape[0]
            need_d21=kwargs.get('need_d21',args[5] if len(args)>5 else True)
            k_m07=len(w2)
            bcs_m07=getattr(self,'_birth_counts',None)
            st_m07=self._higher_storage
            meta_m07=(st_m07[0][:len(self._higher_meta)],st_m07[1][:len(self._higher_meta)],st_m07[2][:len(self._higher_meta)])
            _off_m07=(k_m07*(k_m07-1))//2
            _da_m07=getattr(self,'_higher_diag_all',None)
            _ta_m07=getattr(self,'_higher_traces_all',None)
            _pb_m07=getattr(self,'_higher_proj_buf',None)
            self._higher_last=source_marginals(
                A[:,:ns],P[:,:ns],meta_m07,bufs,
                bcs_m07[:k_m07] if (bcs_m07 and len(bcs_m07)>=k_m07) else None,
                None if _da_m07 is None else _da_m07[_off_m07:_off_m07+k_m07],
                None if _ta_m07 is None else _ta_m07[_off_m07:_off_m07+k_m07],
                None if _pb_m07 is None else _pb_m07[:k_m07],
            )
        return result
    def setup(self,ctx):
        setup0(self,ctx)
        _hd=f.load(str(Path(__file__).with_name('data_01.npz')))
        self.HIGHER_COEFFICIENTS=_hd['coefficients']
        # Optional per-layer recalibration of the deferred response (Claude, 18 Oct; fitted on fresh networks).
        self._higher_fresh_scale=_hd['fresh_scale'].astype(f.float32) if 'fresh_scale' in _hd else None
        prepare_weights(self,getattr(ctx,'depth',16))
        prepare_deferred(self,getattr(ctx,'depth',16))
        n_ctx=getattr(ctx,'width',1024);d_ctx=getattr(ctx,'depth',16)
        if n_ctx>=64 and d_ctx>=2:
            _ensure_higher_buffers(self, n_ctx, d_ctx)
        self.MARGINAL_MODE='predict'
    cls._dslices=wrapped;cls._higher_observe=higher_observe;cls._finish_response=finish_response;cls.setup=setup
    cls.MARGINAL_MODE='off'
