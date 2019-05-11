import numpy as np
import pytest
import root_finding as rf

t1 = (lambda x: x + 3, [-5, 5], -3) #Function has one root, root within interval
t2 = (lambda x: x**2 - 4, [0,5], 2) #Function has two roots, one root within interval
#t3 = (lambda x: x**2 - 4, [-5, 0], -2)

test_cases = [t1, t2]

@pytest.mark.parametrize('f, bounds, expected', test_cases)
def test_bisection_method(f, bounds, expected):
    r = rf.bisection_method(f,bounds)
    
    assert(np.isclose(r, expected))
    
    
@pytest.mark.parametrize('f, bounds, expected', test_cases)
def test_false_position_method(f, bounds, expected):
    r = rf.false_position_method(f,bounds)
    
    assert(np.isclose(r, expected))