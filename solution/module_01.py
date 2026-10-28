"""Same graph contraction in float32, with bounded reusable matrix storage."""
import flopscope.numpy as f

def response(alpha,sigma,d3,g4,covariance,d21,theta,workspace=None):
    n=alpha.shape[0]
    if workspace is None:workspace={}
    if workspace.get('n')!=n:
        workspace.clear();workspace['n']=n
        for key in ('R','T','M'):workspace[key]=f.empty((n,n),dtype=f.float32)
        workspace['message']=f.empty((n,3),dtype=f.float32)
        workspace['part']=f.empty((n,3),dtype=f.float32)
    R,T,M=(workspace[k] for k in ('R','T','M'))
    message,part=workspace['message'],workspace['part']
    a=f.clip(alpha.astype(f.float32),-6.,6.);s=sigma.astype(f.float32);v=s*s
    k3=f.clip(d3.astype(f.float32)/(v*s),-4.,4.)
    k4=f.clip(g4.astype(f.float32)/(v*v),-4.,4.)
    phi=f.exp(-0.5*a*a)*0.3989422804014327
    a2=a*a;a3=a2*a
    H=f.stack((f.ones_like(a),a,a2-1.,a3-3.*a,a2*a2-6.*a2+3.,a2*a3-10.*a3+15.*a),axis=1)
    powers=f.stack((f.ones_like(a),k3,k4,k3*k3,k3*k4,k4*k4),axis=1)
    value=f.sum((H@theta[:36].reshape(6,6))*powers,axis=1)
    nodes=f.stack((f.ones_like(a),a,a2-1.,k3,k4,phi,a*k3),axis=1)
    f.divide(covariance,s[:,None],out=R);f.divide(R,s[None,:],out=R);f.fill_diagonal(R,0.)
    f.divide(d21,v[:,None],out=T);f.divide(T,s[None,:],out=T);f.fill_diagonal(T,0.)
    coefficients=theta[36:].reshape(8,3,7)
    encoded=nodes[None]@f.swapaxes(coefficients,1,2)
    normalization=f.sqrt(f.asarray(n,dtype=f.float32))
    f.matmul(R,encoded[0],out=message)
    for i,matrix in ((1,T),(2,T.T)):
        f.matmul(matrix,encoded[i],out=part);f.add(message,part,out=message)
    for i,left,right in ((3,R,R),(4,R,T),(5,R,T.T),(6,T,T),(7,T,T.T)):
        f.multiply(left,right,out=M)
        f.matmul(M,encoded[i],out=part);f.add(message,part,out=message)
    value=value+f.sum(H[:,:3]*message,axis=1)/normalization
    return (s*phi)*value
