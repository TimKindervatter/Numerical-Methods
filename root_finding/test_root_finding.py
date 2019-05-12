import numpy as np
import pytest
import root_finding as rf

t1 = (lambda x: x + 3, [-5, 5], -3) #Function has one root, root within interval
t2 = (lambda x: x**2 - 4, [0,5], 2) #Function has two roots, one root within interval
t3 = (lambda x: (x-2)**2, [1,4], 2) #Double root at x = 2
t4 = (lambda x: x**4 + 4, [-5, 5], 0) #Function has no roots, expected to fail
test_cases = [t1, t2, t3, pytest.param(*t4, marks=pytest.mark.xfail)]

@pytest.mark.parametrize('f, bounds, expected', test_cases)
def test_bisection_method(f, bounds, expected):
    r = rf.bracketing_method(f,bounds,'bisection')
    
    assert(np.isclose(r, expected))
    
    
@pytest.mark.parametrize('f, bounds, expected', test_cases)
def test_false_position_method(f, bounds, expected):
    r = rf.bracketing_method(f,bounds,'false position')
    
    assert(np.isclose(r, expected))