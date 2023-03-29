# -*- coding: utf-8 -*-
import numpy as np
 


def estimateGrad(fun,x,delta):
     """ Given a real-valued function fun, estimate its gradient numerically.
     """
     grad = SparseVector({})
     for key in x:
         e = SparseVector({})
         e[key] = 1.0
         grad[key] = (fun(x+delta * e) - fun(x))/delta
     return grad

 


