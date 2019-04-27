import numpy as np

def ref(A,b):
    if type(A) is not np.ndarray:
        A = np.array(A)
    if type(b) is not np.ndarray:
        b = np.array(b)
        
    aug = np.ndarray.astype(np.c_[A,b],float)
    for row in range(1,aug.shape[0]):
        for col in range(0,row):
            aug[row] = aug[row] -(aug[row,col]/aug[col,col])*aug[col]
            
    return aug


def back_sub(ref_matrix):
    m = ref_matrix.shape[0]
    x = np.empty([1,m])
    for row in reversed(range(m)):
        signs = -np.ones([1,m])
        signs[0,row] = -signs[0,row]
        ref_matrix[row,:-1] = signs*ref_matrix[row,:-1]
        
        temp_sum = 0
        for i in range(row+1,m):
            temp_sum += ref_matrix[row,i]*x[0,i]
            
        x[0,row] = (temp_sum + ref_matrix[row,-1])/ref_matrix[row,row]
        
    return x


def gaussian_elimination(A,b):
    assert(np.linalg.det(A) != 0)
    
    row_echelon_matrix = ref(A,b)
    x = back_sub(row_echelon_matrix)
    
    return x


def gauss_jordan(A,b=None):
    if type(A) is not np.ndarray:
        A = np.array(A)
        
    m = A.shape[0]
    
    if b is None:
        b = np.identity(m)
    else:
        if type(b) is not np.ndarray:
            b = np.array(b)
        
    aug = np.c_[A,b]
    for row in range(m):
        aug[row] = aug[row]/aug[row,row]
        for i in range(m):
            if i != row:
                aug[i] -= aug[i,row]*aug[row]
            
    if np.allclose(b, np.identity(m)):
        return aug[:,-m:]
    else:
        return aug[:,-1]
    
                