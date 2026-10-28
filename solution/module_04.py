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

"""An effective cubic source using the available covariance and repeated third slice.

This is a defined approximation adapted from the other PC's effective-edge
family. It replaces thin feedback with information folded into the retained
dense leg. It is not an exact re-expression of Rank84's projected feedback.
The missing K22 edge and negative-square completion remain explicitly omitted.
"""
import flopscope.numpy as f

def effective_leg(A,C,h2,alpha,sigma,g,D21,workspace,c_sq=None,r0=0,r1=None):
    """A starts as diag(g) C_off; leave all inputs except A/workspace unchanged."""
    if c_sq is None:
        f.multiply(C,C,out=workspace)
        f.multiply(workspace,h2[:,None],out=workspace)
    else:
        f.multiply(c_sq,h2[:,None],out=workspace)
    ratio=-alpha/(2*sigma)
    f.multiply(workspace,ratio[None,:],out=workspace)
    f.add(A,workspace,out=A)
    if D21 is not None:
        n=A.shape[0]
        if r1 is None:r1=n
        if r0>0 or r1<n:
            d21_s=D21[r0:r1,:r1]
            ws1=workspace[:r1-r0,:r1]
            f.multiply(d21_s,(0.5*h2[r0:r1])[:,None],out=ws1)
            f.add(A[r0:r1,:r1],ws1,out=A[r0:r1,:r1])
            ws2=workspace[:r1,:r1-r0]
            f.multiply(d21_s.T,g[:r1,None],out=ws2)
            f.multiply(ws2,ratio[None,r0:r1],out=ws2)
            f.add(A[:r1,r0:r1],ws2,out=A[:r1,r0:r1])
        else:
            f.multiply(D21,(0.5*h2)[:,None],out=workspace)
            f.add(A,workspace,out=A)
            f.multiply(D21.T,g[:,None],out=workspace)
            f.multiply(workspace,ratio[None,:],out=workspace)
            f.add(A,workspace,out=A)
    f.fill_diagonal(A,0.0)
    return A
