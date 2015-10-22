
"""
The python code is ported from libsvm
"""

from math import exp, log

"""
Platt's binary SVM Probablistic Output: an improvement from 

@article{note-platt-jml07,
 author = {Lin, Hsuan-Tien and Lin, Chih-Jen and Weng, Ruby},
 title = {A note on {Platt's} probabilistic outputs for support vector machines},
 journal = {Mach. Learn.},
 volume = {68},
 issue = {3},
 year = {2007},
 pages = {267-276},
 numpages = {10},
} 

"""

def sigmoid_train(dec_values, labels, verbose=0):
    assert(len(dec_values) == len(labels))
    n = len(dec_values)
   
    prior1 = 0.0
    prior0 = 0.0

    for i in range(n):
        if labels[i] > 0:
            prior1 += 1
        else:
            prior0 += 1
	
    max_iter = 100         # Maximal number of iterations
    min_step = 1e-10       # Minimal step taken in line search
    sigma = 1e-12          # For numerically strict PD of Hessian
    eps = 1e-5
    hiTarget = (prior1+1.0) / (prior1+2.0)  # Eq. 2 in Lin et al.
    loTarget = 1 / (prior0+2.0)

    t = [0.0] * n
	
    # Initial Point and Initial Fun Value
    A = 0.0
    B = log((prior0+1.0)/(prior1+1.0))
    fval = 0.0

    for i in range(n):
        if (labels[i]>0):
            t[i] = hiTarget
	else:
            t[i] = loTarget

        fApB = dec_values[i]*A + B
        if (fApB>=0):
            fval += t[i]*fApB + log(1+exp(-fApB))
	else:
            fval += (t[i]-1)*fApB +log(1+exp(fApB))

    if verbose:
        print "[sigmoid_train] iter %d, A %g, B %g, fval %g" % (0, A, B, fval)

    for iter in range(max_iter):
       
        # Update Gradient and Hessian (use H' = H + sigma I)
        h11 = sigma # numerically ensures strict PD
	h22 = sigma
        h21 = 0.0
        g1  = 0.0
        g2  = 0.0

	for i in range(n):
            fApB = dec_values[i]*A + B

            if (fApB >= 0):
                p = exp(-fApB) / (1.0+exp(-fApB))
                q = 1.0 / (1.0+exp(-fApB))
            else:
                p = 1.0 / (1.0+exp(fApB))
                q = exp(fApB) / (1.0+exp(fApB))
			
            d2 = p * q
            h11 += dec_values[i]*dec_values[i]*d2
            h22 += d2
            h21 += dec_values[i]*d2
            d1 = t[i] - p
            g1 += dec_values[i]*d1
            g2 += d1

        # Stopping Criteria
        if (abs(g1)<eps and abs(g2)<eps):
            #print "[sigmoid_train] hit stopping criteria."
            break

        # Finding Newton direction: -inv(H') * g
        det = h11*h22 - h21*h21
        dA =- (h22*g1 - h21 * g2) / det
        dB =- (-h21*g1+ h11 * g2) / det
        gd = g1*dA + g2*dB

        stepsize = 1.0 # Line Search
        while (stepsize >= min_step):
            newA = A + stepsize * dA
            newB = B + stepsize * dB

            # New function value
            newf = 0.0

            for i in range(n):
                fApB = dec_values[i]*newA + newB

                if (fApB >= 0):
                    newf += t[i]*fApB + log(1+exp(-fApB))
                else:
                    newf += (t[i] - 1)*fApB +log(1+exp(fApB))

            # Check sufficient decrease
            if (newf<fval+0.0001*stepsize*gd):
                A = newA
                B = newB
                fval = newf
                break
            else:
                stepsize = stepsize / 2.0
        # end of Line Search

        if (stepsize < min_step):
            print "[sigmoid_train] Line search fails in two-class probability estimates"
            break

        if verbose:
            print "[sigmoid_train] iter=%d, abs(g1)=%g, abs(g2)=%g, A=%g, B=%g, fval=%g" % (iter+1, abs(g1), abs(g2), A, B, fval)
    # end of iteration

    if (iter >= (max_iter-1)):
        print "[sigmoid_train] Reaching maximal iterations in two-class probability estimates"

    return [A, B]


def sigmoid_predict(decision_value, A, B):
    fApB = decision_value*A+B
    if (fApB >= 0):
        return exp(-fApB)/(1.0+exp(-fApB))
    else:
        return 1.0/(1+exp(fApB))



if __name__ == "__main__":
    A = -0.1
    B = 0

    [A, B] = sigmoid_train(dec_values=[2, 1, -1, -2, 1], labels=[1, 1, -1, -1, -1], verbose=1)
    print A, B

    for x in [-1, 0, 1, 10, 100, 120]:
        print x, sigmoid_predict(x, -01, 0), sigmoid_predict(x, A, B)


