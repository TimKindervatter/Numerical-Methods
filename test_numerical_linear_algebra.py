import numerical_linear_algebra as nla
import numpy as np

def test_gaussian_elimination():
    A = [[10,3,9], [7,3,5], [0,4,7]]
    b = [1,1,1]
    
    x1 = np.linalg.solve(A,b)
    x2 = nla.gaussian_elimination(A,b)
    
    assert(np.all(np.isclose(x1,x2)))