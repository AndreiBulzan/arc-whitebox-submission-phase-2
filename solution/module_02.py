"""The frozen float32 graph, evaluated independently across a layer batch."""
import flopscope.numpy as f
from _zero_store import zero32 as _zero32


def reserve(workspace, layers, width):
    if workspace.get('geometry') != (layers, width):
        workspace.clear()
        workspace['geometry'] = (layers, width)
        for key in ('R', 'T', 'M'):
            workspace[key] = f.empty((layers, width, width), dtype=f.float32)
        for key in ('R', 'T'):
            workspace[f'{key}_diagonal'] = f.reshape(workspace[key], (layers, width*width))[:, ::width+1]
        for key in ('message', 'part'):
            workspace[key] = f.empty((layers, width, 3), dtype=f.float32)


def response(alpha, sigma, d3, g4, covariance, d21, theta, workspace):
    layers, n = alpha.shape
    reserve(workspace, layers, n)
    R, T, M = (workspace[k] for k in ('R', 'T', 'M'))
    message, part = workspace['message'], workspace['part']
    a = f.clip(alpha.astype(f.float32), -6., 6.)
    s = sigma.astype(f.float32)
    v = s*s
    k3 = f.clip(d3.astype(f.float32)/(v*s), -4., 4.)
    k4 = f.clip(g4.astype(f.float32)/(v*v), -4., 4.)
    phi = f.exp(-0.5*a*a)*0.3989422804014327
    a2 = a*a
    a3 = a2*a
    H = f.stack((f.ones_like(a), a, a2-1., a3-3.*a,
                 a2*a2-6.*a2+3., a2*a3-10.*a3+15.*a), axis=2)
    powers = f.stack((f.ones_like(a), k3, k4, k3*k3, k3*k4, k4*k4), axis=2)
    value = f.sum((H@theta[:, :36].reshape(layers, 6, 6))*powers, axis=2)
    nodes = f.stack((f.ones_like(a), a, a2-1., k3, k4, phi, a*k3), axis=2)
    # covariance/d21 may be these owned slabs themselves after capture.
    f.divide(covariance, s[:, :, None], out=R)
    f.divide(R, s[:, None, :], out=R)
    f.divide(d21, v[:, :, None], out=T)
    f.divide(T, s[:, None, :], out=T)
    f.copyto(workspace['R_diagonal'], _zero32())
    f.copyto(workspace['T_diagonal'], _zero32())
    coefficients = theta[:, 36:].reshape(layers, 8, 3, 7)
    encoded = nodes[:, None]@f.swapaxes(coefficients, 2, 3)
    normalization = f.sqrt(f.asarray(n, dtype=f.float32))
    f.matmul(R, encoded[:, 0], out=message)
    transposed = f.swapaxes(T, 1, 2)
    for i, matrix in ((1, T), (2, transposed)):
        f.matmul(matrix, encoded[:, i], out=part)
        f.add(message, part, out=message)
    for i, left, right in ((3, R, R), (4, R, T), (5, R, transposed),
                           (6, T, T), (7, T, transposed)):
        f.multiply(left, right, out=M)
        f.matmul(M, encoded[:, i], out=part)
        f.add(message, part, out=message)
    value = value+f.sum(H[:, :, :3]*message, axis=2)/normalization
    return (s*phi)*value
