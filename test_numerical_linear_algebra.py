import pytest
import numerical_linear_algebra as nla
import numpy as np

A1 = np.array([[10,3,9], [7,3,5], [0,4,7]]) #Non-singular ndarray
A2 = [[4,2,6,2], [6,7,2,1], [3,7,1,4], [1,6,2,8]] #Non-singular list of lists
A3 = np.identity(10)
A4 = np.zeros([2,2]) #Singular, expected to fail

test_cases = [A1, A2, A3, pytest.param(A4, marks=pytest.mark.xfail(strict=True))]

#Randomized testing
n = np.random.randint(1,100)
random_tests = [np.random.randn(n,n) for _ in range(10)]

all_tests = test_cases + random_tests


@pytest.mark.parametrize('A', all_tests)
def test_ref(A):
    b = np.ones(len(A))
    L,U,_ = nla.ref(A,b)
    
    assert(np.allclose(L@U, A))
    
  
@pytest.mark.parametrize('A', all_tests)
def test_forward_sub(A):
    b = np.ones(len(A))
    L,_,_ = nla.ref(A,b)
    
    try:
        d1 = np.linalg.solve(L,b)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            d1 = np.full(len(L), np.nan)
            
    d2 = nla.forward_sub(L,b)
    
    assert(np.allclose(d1,d2))
    
    
@pytest.mark.parametrize('A', all_tests)
def test_back_sub(A):
    b = np.ones(len(A))
    _,U,_ = nla.ref(A,b)
    
    try:
        x1 = np.linalg.solve(U,b)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            x2 = np.full(len(U), np.nan)
            
    x2 = nla.back_sub(U,b)
    
    assert(np.allclose(x1,x2))


@pytest.mark.parametrize('A', all_tests)
def test_gaussian_elimination(A):
    b = np.ones(len(A))
    
    try:
        x1 = np.linalg.solve(A,b)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            x1 = np.full(len(A), np.nan)
            
    x2 = nla.gaussian_elimination(A,b)
    
    assert(np.allclose(x1,x2))
    
    
@pytest.mark.parametrize('A', all_tests)
def test_gauss_jordan_solve(A):
    b = np.ones(len(A))
    
    try:
        x1 = np.linalg.solve(A,b)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            x1 = np.full(len(A), np.nan)
            
    x2 = nla.gauss_jordan(A,b)
    
    assert(np.allclose(x1,x2))
    
    
@pytest.mark.parametrize('A', all_tests)
def test_gauss_jordan_invert(A):
    if type(A) is not np.ndarray:
        A = np.array(A)
        
    m = A.shape[0]
    A_inv = nla.gauss_jordan(A)
    
    assert(np.all(np.isclose(A_inv@A, np.identity(m))))
    assert(np.all(np.isclose(A@A_inv, np.identity(m))))
    

@pytest.mark.parametrize('A', all_tests)
def test_lu(A):
    b = np.ones(len(A))
    
    try:
        x1 = np.linalg.solve(A,b)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            x1 = np.full(len(A), np.nan)
            
    x2 = nla.lu(A,b)
    
    assert(np.allclose(x1,x2))