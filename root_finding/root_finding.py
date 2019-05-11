import numpy as np

class RootNotFoundError(Exception):
    """Raised if the sign of the function does not change in the specified interval."""
    def __init__(self, message):
        super().__init__(message)
    
    
def bisection_method(f,bounds):
    bisection = lambda xl, xu: (xl + xu)/2
    return bracketing_method(f, bounds, bisection)


def false_position_method(f,bounds):
    def false_position(xl, xu):
        print(f(xu) - f(xl))
        return (xl + xu)/2 - (((f(xu)) + f(xl))/(f(xu) - f(xl)))*((xu - xl)/2)
    return bracketing_method(f, bounds, false_position)


def bracketing_method(f, bounds, iteration_method):
    """
    Finds the roots of the supplied function. The interval to search within, as well as the method by which the root estimate is updated, must be provided.
    
    Args:
        f (function): The function whose roots are to be evaluated.
        bounds (list): A two-item list containing the lower and upper bounds of the interval in which to search for a root.
        iteration_method (function): The function to be used for iterating the root estimate.
        
    Returns:
        A root of the function f, if one was found within the supplied interval.
    """
    
    xl = bounds[0]
    xu = bounds[1]
    
    delta_x = 1
    
    while delta_x > 1e-10:
        x_old = iteration_method(xl, xu)
        
        if f(xl)*f(x_old) < 0:
            xu = x_old
        elif f(xu)*f(x_old) < 0:
            xl = x_old
        elif f(x_old) == 0:
            return x_old
        else:
            raise RootNotFoundError("The sign did not change in the given interval. This indicates either no root or an even root.")
            
        xnew = iteration_method(xl, xu)
        
        delta_x = abs((xnew - x_old)/(x_old + 1e-10))
        
    return xnew