"""Young quartic response sharing the parent cubic query's A/P right factors."""
import flopscope.numpy as f

def left_polynomials(a,p,coefficients,pp=None,scale=None):
    """Four physical families: 4A³P, 6A²P², 4AP³, P⁴ (reusing PP and pre-scaling 1D coefficients)."""
    c3 = 3.0 * coefficients[:, :3, None, :]
    l2, l3, l4, l5 = (coefficients[:, i, None, :] for i in range(4))
    l2_3, l3_3, l4_3 = c3[:, 0], c3[:, 1], c3[:, 2]
    if pp is None:
        pp = p * p
    al2 = a * l2
    pl4 = p * l4
    pl3 = p * l3_3
    if scale is not None:
        sa = a * scale
        s_aa = sa * a
        s_ap = sa * p
        s_pp = pp * scale
        ka = s_ap * (a * l2_3 + pl3) + s_pp * pl4
        kp = s_aa * (al2 + pl3) + s_pp * (a * l4_3 + p * l5)
        return ka, kp
    aa = a * a; ap = a * p
    ka = ap * (a * l2_3 + pl3) + pp * pl4
    kp = aa * (al2 + pl3) + pp * (a * l4_3 + p * l5)
    return ka, kp

class LiveFold:
    def __init__(self,natural_coefficients,max_age=3):
        self.natural=natural_coefficients;self.max_age=max_age
        self.birth=[];self.rows=[];self.layer=0
    def before(self,layer,order,alpha,sigma):
        if layer==0:self.birth=[];self.rows=[]
        self.layer=layer;self.scale=-alpha/(3.*sigma)
        value=self.natural.get(layer)
        if value is None:value=f.zeros((4,order.shape[0]),dtype=f.float32)
        self.birth.append(f.take(value,order,axis=1))
    def augment(self,la,lp,a,p,pp,ka,start,nr,nq,n):
        k,ns,_=a.shape
        assert k==self.layer,(k,self.layer)
        lo=max(ka,k-self.max_age)
        if lo>=k:return None
        bcs=getattr(self,'birth_counts',None)
        if bcs is not None and len(bcs)>=k:
            groups=[]
            cur=lo
            while cur<k:
                end=cur+1;bc=bcs[cur]
                while end<k and bcs[end]==bc:end+=1
                groups.append((cur,end,bc));cur=end
        else:
            groups=[(lo,k,a.shape[2])]
        diagonal=None
        sc=self.scale[None,:ns,None]
        for g_lo,g_hi,bc in groups:
            a_s=a[g_lo:g_hi,:,:bc];p_s=p[g_lo:g_hi,:,:bc];pp_s=pp[g_lo:g_hi,:,:bc]
            coeff=f.stack(self.birth[g_lo:g_hi],axis=0)[:,:,:bc]
            lefta,leftp=left_polynomials(a_s,p_s,coeff,pp=pp_s,scale=sc)
            diag_g=f.einsum('kij,kij->i',lefta,a_s)+f.einsum('kij,kij->i',leftp,p_s)
            diagonal=diag_g if diagonal is None else (diagonal+diag_g)
            f.add(la[g_lo:g_hi,:,:bc],lefta,out=la[g_lo:g_hi,:,:bc])
            f.add(lp[g_lo:g_hi,:,:bc],leftp,out=lp[g_lo:g_hi,:,:bc])
        full=f.pad(diagonal,(0,n-ns),mode='constant') if ns<n else diagonal
        visible=f.zeros(n,dtype=full.dtype)
        hi=min(start+nr,nq,ns)
        if hi>start:f.copyto(visible[start:hi],full[start:hi])
        return visible,full

"""Dimensionless birth features with the quartic families' uniform scale law."""
import flopscope as fs
import flopscope.numpy as f

def dimensionless_features(alpha, sigma, d21_diag, g4, a, p, ones=None, return_variance=False):
    # The live controller already owns float64 birth vectors.
    # All expressions below are read-only with respect to these inputs.
    al = alpha if alpha.dtype == f.float64 else alpha.astype(f.float64)
    sg = sigma if sigma.dtype == f.float64 else sigma.astype(f.float64)
    d = d21_diag if d21_diag.dtype == f.float64 else d21_diag.astype(f.float64)
    q = g4 if g4.dtype == f.float64 else g4.astype(f.float64)
    av, pv = a.astype(f.float64), p.astype(f.float64)
    aa, pp = av*av, pv*pv
    na = f.sqrt(f.sum(aa, axis=0))+1e-12
    np_ = f.sqrt(f.sum(pp, axis=0))+1e-12
    overlap = f.einsum('ij,ij->j', aa, pp)+1e-24
    phi = f.exp(-.5*al*al)*0.3989422804014327
    gate = fs.stats.norm.cdf(al)
    sg2 = sg*sg
    sg4 = sg2*sg2
    al2 = al*al
    features = f.stack((f.ones_like(al) if ones is None else ones, al, gate, phi, al*phi,
                    (al2-1.)*phi, al2,
                    d/(sg2*sg),
                    q/sg4,
                    f.log(na/sg2), f.log(np_), f.log(overlap/sg4)), axis=0)
    return (features, sg2) if return_variance else features

class ScaleLawRuleFold(LiveFold):
    def __init__(self, weights, scale, families, depth, max_age=3):
        super().__init__({}, max_age=max_age)
        assert tuple(families) == (2, 3, 4, 5)
        assert weights.shape == (4, 3, 36) and scale.shape == (12,) and max_age == 3
        self.weights = weights.astype(f.float64)
        self.feature_scale = scale.astype(f.float64)
        self._weight_matrix = self.weights.reshape(12, 36)
        self._scale_column = self.feature_scale[:, None]
        self._time_depth = None
        self._ones_width = None
        self._zeros_f64 = None
        self.depth = depth
        if depth == 16:
            self._birth_ones = f.ones(1024, dtype=f.float64)
            self._ones_width = 1024
            self._birth_times = (f.arange(max(0, depth-2), dtype=f.float64)
                                 / f.asarray(max(1, depth-2), dtype=f.float64))
            self._time_depth = depth
            self._zeros_f64 = f.zeros(1024, dtype=f.float64)

    def before(self, layer, order, alpha, sigma):
        if layer == 0:
            self.birth = []
            self.rows = []
            self.base = []
            self.moments = []
            self.future = []
            self.zero = None
            if self._ones_width != order.shape[0]:
                self._birth_ones = f.ones(order.shape[0], dtype=f.float64)
                self._ones_width = order.shape[0]
            if self._time_depth != self.depth:
                self._birth_times = (f.arange(max(0, self.depth-2), dtype=f.float64)
                                     / f.asarray(max(1, self.depth-2), dtype=f.float64))
                self._time_depth = self.depth
        self.layer = layer
        self.scale = -alpha/(3.*sigma) if 1 <= layer < self.depth - 1 else None
        # Own the birth coordinates; parent arrays are reused later.
        self.base.append((alpha.astype(f.float64), sigma.astype(f.float64))
                         if layer+2 < self.depth else None)
        self.moments.append(None)
        self.future.append(None)

    def observe_birth(self, layer, d21, g4):
        if layer+2 >= self.depth:
            return
        n = self.base[layer][0].shape[0]
        if layer == 0:
            z64 = self._zeros_f64 if (self._zeros_f64 is not None and self._zeros_f64.shape[0] == n) else f.zeros(n, dtype=f.float64)
            d = z64
            q = z64
        else:
            d = f.diag(d21).astype(f.float64)
            q = g4.astype(f.float64)
        self.moments[layer] = (d, q)

    def first_consumer(self, a, p):
        layer, g = self.layer, self.layer-1
        if a is not None and g >= 0 and g+2 < self.depth:
            assert a.shape[0] == layer and a.shape == p.shape
            n_full = a.shape[2]
            bcs = getattr(self, 'birth_counts', None)
            bc = bcs[g] if (bcs is not None and len(bcs) > g) else n_full
            al, sg = self.base[g][0][:bc], self.base[g][1][:bc]
            d, q = self.moments[g][0][:bc], self.moments[g][1][:bc]
            ones_bc = None if self._birth_ones is None else self._birth_ones[:bc]
            x, sg2 = dimensionless_features(al, sg, d, q, a[-1, :, :bc], p[-1, :, :bc],
                                           ones=ones_bc, return_variance=True)
            x = x/self._scale_column
            t = self._birth_times[g:g+1]
            xt = x*t
            x = f.concatenate((x, xt, xt*t), axis=0)
            future = (self._weight_matrix@x).reshape(4, 3, -1)
            powers = f.stack((1./sg2, ones_bc if ones_bc is not None else f.ones_like(sg2), sg2, sg2*sg2), axis=0)
            future = future*powers[:, None, :]
            fut32 = f.swapaxes(future, 0, 1).astype(f.float32)
            if bc < n_full:
                fut32 = f.pad(fut32, ((0, 0), (0, 0), (0, n_full - bc)), mode='constant')
            self.future[g] = fut32
        self.birth = [self.zero]*(layer+1)
        for age in range(1, min(self.max_age, layer)+1):
            bank = self.future[layer-age]
            if bank is not None:
                self.birth[layer-age] = bank[age-1]


from pathlib import Path
from module_08 import general_response

def install(cls):
    setup_parent = cls.setup
    predict_parent = cls.predict
    def setup(self, ctx):
        setup_parent(self, ctx)
        data = f.load(str(Path(__file__).with_name('data_04.npz')))
        self._fold_controller = ScaleLawRuleFold(
            data['w'], data['scale'], (2, 3, 4, 5), getattr(ctx, 'depth', 0), max_age=3)
    def predict(self, mlp, budget):
        # The fitted recurrence and response were jointly calibrated at this
        # architecture. Use an analytic covariance closure on other shapes.
        if (mlp.width, mlp.depth) != (1024, 16):
            return general_response(mlp)
        self._fold_controller.depth = mlp.depth
        return predict_parent(self, mlp, budget)
    cls.setup = setup
    cls.predict = predict
