import flopscope.numpy as f

def padding_mask(owner,width,widths):
    cache=getattr(owner,'_ragged_window_masks',None)
    if cache is None:
        cache={};owner._ragged_window_masks=cache
    key=(width,widths)
    result=cache.get(key)
    if result is None:
        s=getattr(owner,'_arange_slices_n',None)
        idx=s[1].get((0,width)) if (s is not None and (0,width) in s[1]) else f.arange(width,dtype=f.int32)
        result=idx[None,:]>=f.asarray(widths,dtype=f.int32)[:,None]
        if len(cache)>=128:cache.pop(next(iter(cache)))
        cache[key]=result
    return result

def scatter(owner,packed,out,ns,widths):
    zero=getattr(owner,'_ragged_zero32',None)
    if zero is None:
        zero=f.zeros((),dtype=f.float32);owner._ragged_zero32=zero
    total=sum(widths);width=out.shape[3];pad=width-widths[-1]
    windows=[];offset=0
    n_full=len(widths) if pad==0 else len(widths)-1
    for bc in widths[:n_full]:
        windows.append(packed[:,offset:offset+width]);offset+=bc
    if n_full:
        f.stack(tuple(windows),axis=0,out=out[:n_full,0,:ns,:])
        f.copyto(out[:n_full,0,:ns,:],zero,where=padding_mask(owner,width,widths[:n_full])[:,None,:])
    if pad:
        bc_last=widths[-1]
        f.copyto(out[n_full,0,:ns,:bc_last],packed[:,offset:offset+bc_last])
        f.copyto(out[n_full,0,:ns,bc_last:],zero)
    if ns<out.shape[2]:f.copyto(out[:,:,ns:,:],zero)

def install(cls):
    previous=cls._young_mm
    def young(self,smm,X,Y,out,lev,lo,hi):
        counts=self._birth_counts[lo:hi]
        if min(counts)==max(counts):return previous(self,smm,X,Y,out,lev,lo,hi)
        widths=tuple(bc for bc in counts for _ in range(2))
        total=sum(widths);kd=X.shape[-1]
        ns=out.shape[2] if self._output_ids is None else self._output_ids.shape[0]
        rhs=smm._arena('root',('R',1),(kd,total))
        rv=smm._arena('root',('R',1),(1,1,kd,total))
        packed=smm._arena('root',('O',1),(ns,total))
        ov=smm._arena('root',('O',1),(1,1,ns,total))
        f.concatenate(tuple(Y[i,0,:,:bc] for i,bc in enumerate(widths)),axis=1,out=rhs)
        smm.mm(X if ns == X.shape[2] else smm._execution_slice(X, 2, None, ns),rv,ov,lev)
        scatter(self,packed,out,ns,widths)
    cls._young_mm=young
