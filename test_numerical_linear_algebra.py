import pytest
import numerical_linear_algebra as nla
import numpy as np

A1 = [[10,3,9], [7,3,5], [0,4,7]]
A2 = [[4,2,6,2], [6,7,2,1], [3,7,1,4], [1,6,2,8]]
A3 = np.identity(10)
A4 = np.zeros([2,2])
test_cases = [A1, A2, A3, pytest.param(A4, marks=pytest.mark.xfail(strict=True))]

#Randomized testing
n = np.random.randint(1,10)
random_tests = [np.random.randn(n,n) for _ in range(10)]

@pytest.mark.parametrize('A', test_cases)
def test_gaussian_elimination(A):
    b = np.ones(len(A))
    
    try:
        x1 = np.linalg.solve(A,b)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            x1 = np.full(len(A), np.nan)
    x2 = nla.gaussian_elimination(A,b)
    
    assert(np.all(np.isclose(x1,x2)))
    
    
def test_gauss_jordan():
    return True