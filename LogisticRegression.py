# -*- coding: utf-8 -*-
import numpy as np
import argparse
import warnings
from time import time
from SparseVector import SparseVector


def readBeta(input):
    """ Read a vector β from file input. Each line contains pairs of the form:
                (feature,value)
    """
    beta = SparseVector({})
    with open(input,'r') as fh:
        for  line in fh:
            (feat,val) = eval(line.strip())
            beta[feat] = val
    return beta

def writeBeta(output,beta):
    """ Write a vector β to a file output.  Each line contains pairs of the form:
                (feature,value)
 
    """
    with open(output,'w') as fh:
        for key in beta:
            fh.write('(%s,%f)\n' % (key,beta[key]))

def readData(input_file):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The result is stored in a list containing tuples of the form
                 (SparseVector(x),y)             

    """ 
    listSoFar = []
    with open(input_file,'r') as fh:
        for line in fh:
            (x,y) = eval(line)
            x = SparseVector(x)
            listSoFar.append((x,y))

    return listSoFar

def getAllFeatures(data):
    """ Get all the features present in dataset data.
    """
    features = SparseVector({})
    for (x,y) in data:
        features = features + x
    return features.keys() 

def logisticLoss(beta,x,y):
    """
        Given sparse vector beta, a sparse vector x, and a binary value y in {-1,+1}, compute the logistic loss
               
                l(β;x,y) = log( 1.0 + exp(-y * <β,x>) )

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.log(1. + np.exp (-y * beta.dot(x)))

def gradLogisticLoss(beta,x,y):
    """
        Given a sparse vector beta, a sparse vector x, and 
        a binary value y in {-1,+1}, compute the gradient of the logistic loss 

              ∇l(B;x,y) = -y / (1.0 + exp(y <β,x> )) * x
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return - 1.* y / (1. + np.exp(y*beta.dot(x)) ) * x

def totalLoss(data,beta,lam = 0.0):
    """  Given a sparse vector beta and a dataset  compute the regularized total logistic loss :
              
               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2             
        
         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

         Output is:
            - The loss L(β) 
    """
    loss = 0.0
    for (x,y) in data:
        loss += logisticLoss(beta,x,y)
    return loss + lam * beta.dot(beta)

def gradTotalLoss(data,beta, lam = 0.0):
    """  Given a sparse vector beta and a dataset perform compute the gradient of regularized total logistic loss :
            
              ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β   
        
         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

         Output is:
            - The gradient ∇L(β) 
    """    
    total_loss = SparseVector({})
    for (x, y) in data:
        total_loss += gradLogisticLoss(beta, x, y)
    return total_loss + 2 * lam * beta

def lineSearch(fun,x,grad,a=0.2,b=0.6):
    """ Given function fun, a current argument x, and gradient grad=∇fun(x), 
        perform backtracking line search to find the next point to move to.
        (see Boyd and Vandenberghe, page 464).

        Both x and grad are presumed to be SparseVectors.
        
        Inputs are:
            - fun: the objective function f.
            - x: the present input (a Sparse Vector)
            - grad: the present gradient ∇f(x) (as Sparse Vector)
            - Optional parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient grad=∇fun(x), the function finds a t such that
        fun(x - t * ∇f(x)) <= f(x) - a * t * <∇f(x),∇f(x)>

        The return value is the resulting value of t.
    """
    t = 1.0

    fx = fun(x)
    gradNormSq = grad.dot(grad)

    while fun(x-t*grad) > fx- a * t * gradNormSq:
        t = b * t
    return t 
    
def basicMetrics(data,beta):
    """ Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.
        
        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0
                 TP = datapoints in (x,y) in P for which y=+1  
                 FP = datapoints in (x,y) in P for which y=-1  
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return value is a tuple containing
             - #P,#N,#TP,#FP,#TN,#FN
    """
    pairs = ( ( int(np.sign(beta.dot(x))), int(y)) for (x,y) in data  )
    new_pairs = [ (pred_label,pred_label*true_label)  for (pred_label,true_label) in pairs ]        
    

    TP = 1.*new_pairs.count( (1,1) )
    FP = 1.*new_pairs.count( (1,-1) )
    TN = 1.*new_pairs.count( (-1,1) )
    FN = 1.*new_pairs.count( (-1,-1) )
    P = TP+FP
    N = TN+FN
    return P,N,TP,FP,TN,FN 

def metrics(P,N,TP,FP,TN,FN):
    """Regurn the accuracy (ACC), precision (PRE), and recall (REC). These are defined in terms of the following sets:

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall are defined as:
         
                 ACC = ( #TP+#TN ) / (#P + #N)
                 PRE = #TP / (#TP + #FP)
                 REC = #TP/ (#TP + #FN)

        Inputs are:
             - #P,#N,#TP,#FP,#TN,#FN

        The return value is a tuple containing
             - ACC, PRE, REC
    """
    acc = (TP+TN)/(P+N)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
 
    return acc,pre,rec

def test(data,beta):
    """Return the accuracy (ACC), precision (PRE), and recall (REC) of β over dataset data.

       Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return value is a tuple containing
             - ACC, PRE, REC
    """
    P,N,TP,FP,TN,FN = basicMetrics(data,beta)
    return metrics(P,N,TP,FP,TN,FN)

def train(data,beta_0, lam,max_iter,eps,test_data=None):
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:

        grad = gradTotalLoss(data,beta,lam)  

        fun = lambda x: totalLoss(data,x,lam)
        gamma = lineSearch(fun,beta,grad)

        beta = beta - gamma * grad

        obj = fun(beta)   
        gradNorm = np.sqrt(grad.dot(grad))
        if test_data == None:
            print('k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_{k-1})||_2 = ',gradNorm,'\tγ = ',gamma)
        else:
            acc, pre, rec = test(test_data,beta)
   
            print('k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tγ = ',gamma,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec)
        k = k + 1

    return beta,gradNorm,k         


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int,default=10, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='ε-tolerance. If the l2_norm of the gradient is smaller than ε, gradient descent terminates.') 

    test_group = parser.add_mutually_exclusive_group(required=False)
    test_group.add_argument('--online_test', dest='test_while_training', action='store_true',help="Test during training. --testdata must be provided.")
    test_group.add_argument('--end_test_only', dest='test_while_training', action='store_false',help="Suppress testing during training. If --testdata is provided, testing will happen only at the very end of the training.")
    parser.set_defaults(test_while_training=False)
 
    args = parser.parse_args()
    

    print('Reading training data from',args.traindata)
    traindata = readData(args.traindata)
    print('Read',len(traindata),'data points with',len(getAllFeatures(traindata)),'features in total.')
    
    if args.testdata:
        print('Reading test data from',args.testdata)
        testdata = readData(args.testdata)
        print('Read',len(testdata),'data points with',len(getAllFeatures(testdata)),'features in total.')
    else:
        testdata = None

    beta0 = SparseVector({})

    print('Training on data from',args.traindata,'with: λ = %f, ε = %f, max iter= %d:' % (args.lam,args.eps,args.max_iter))
    if args.test_while_training:
        beta, gradNorm, k = train(traindata,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps,test_data=testdata) 
    else:
        beta, gradNorm, k = train(traindata,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps) 
    print('Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps)
    print('Saving trained β in',args.beta)
    writeBeta(args.beta,beta)

    if args.testdata and not args.test_while_training:
       acc, pre, rec = test(testdata,beta)
       print("Final test performance at trained β is:",'ACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec) 
