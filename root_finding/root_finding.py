import numpy as np
from autograd import grad

class RootNotFoundError(Exception):
    """Raised if the sign of the function does not change in the specified interval."""
    def __init__(self, message):
        super().__init__(message)
    
    
def bisection_method(f,bounds):
    return lambda xl, xu: (xl + xu)/2


def false_position_method(f,bounds):
    return lambda xl, xu: (xl + xu)/2 - (((f(xu)) + f(xl))/(f(xu) - f(xl)))*((xu - xl)/2)


def bracketing_method(f, bounds, iteration_method):
    """
    Finds the roots of the supplied function. The interval to search within, as well as the method by which the root estimate is updated, must be provided.
    
    Args:
        f (function): The function whose roots are to be evaluated.
        bounds (list): A two-item list containing the lower and upper bounds of the interval in which to search for a root.
        iteration_method (string): Either 'bisection' or 'false position', to choose the method for updating the root during each iteration.
        
    Returns:
        A root of the function f, if one was found within the supplied interval.
    """
    
    #Choose which method to use based on the iteration_method input argument
    def generate_root_estimator(function):
        """Returns a lambda expression that calculates a new guess for the root."""
        if iteration_method.lower() == 'false position':
            update_root = false_position_method(function,bounds)
        if iteration_method.lower() == 'bisection':
            update_root = bisection_method(function,bounds)
            
        return update_root
    
    
    update_root = generate_root_estimator(f)
    
    xl = bounds[0]
    xu = bounds[1]
    
    delta_x = 1
    
    #Iterate until the new guess differs from the old guess by some small number
    while delta_x > 1e-10:
        #Estimate the root's position
        x_old = update_root(xl, xu)
        
        #If the function changes between the lower bound and the estimate, the root lies between those points
        if f(xl)*f(x_old) < 0:
            xu = x_old
        #If the function changes between the lower bound and the estimate, the root lies between those points
        elif f(xu)*f(x_old) < 0:
            xl = x_old
        #If the value of the function at the estimate is very close to 0, we found a root
        elif np.isclose(f(x_old),0):
            return x_old
        #If no sign change occurred in the interval:
        else:
            try:
                #try recursively calling the root finding method with the auxiliary function f(x)/f'(x)
                u = lambda x: f(x)/grad(f)(x + 1e-10)
                update_root = generate_root_estimator(u)
                return bracketing_method(u, bounds, iteration_method)
            #If this still didn't work, there's no root in the interval
            except:
                raise RootNotFoundError("No root was found within the given interval.")
            
        xnew = update_root(xl, xu)
        
        delta_x = abs((xnew - x_old)/(x_old + 1e-10))
        
    return xnew


if __name__ == '__main__':
    f = lambda x: (x - 2)**2
    bracketing_method(f, [1,4],'false_position')