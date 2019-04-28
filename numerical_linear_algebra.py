import numpy as np

def ref(A,b):
    if type(A) is not np.ndarray:
        A = np.array(A)
    if type(b) is not np.ndarray:
        b = np.array(b)
        
    assert(np.linalg.det(A) != 0)
        
    aug = np.ndarray.astype(np.c_[A,b],float)
    L = np.identity(A.shape[0])
    for row in range(1,aug.shape[0]):
        for col in range(0,row):
            L[row,col] = (aug[row,col]/aug[col,col])
            aug[row] = aug[row] -L[row,col]*aug[col]
            
    U = aug[:,:-1]
    y = aug[:,-1]
    return L, U, y


def back_sub(U, y):
    m = U.shape[0]
    x = np.empty(m)
    ref_matrix = np.c_[U,y]
    
    for row in reversed(range(m)):
        signs = -np.ones([1,m])
        signs[0,row] = -signs[0,row]
        ref_matrix[row,:-1] = signs*ref_matrix[row,:-1]
        
        temp_sum = 0
        for i in range(row+1,m):
            temp_sum += ref_matrix[row,i]*x[i]
            
        x[row] = (temp_sum + ref_matrix[row,-1])/ref_matrix[row,row]
        
    return x


def forward_sub(L,b):
    m = L.shape[0]
    d = np.empty(m)
    ref_matrix = np.c_[L,b]
    
    d[0] = ref_matrix[0,0]
    
    for row in range(1,m):
        temp_sum = b[row]
        for i in range(row):
            temp_sum -= ref_matrix[row,i]*d[i]
            
        d[row] = temp_sum
        
    return d


def gaussian_elimination(A,b):
    assert(np.linalg.det(A) != 0)
    
    _, U, y = ref(A,b)
    x = back_sub(U, y)
    
    return x


def gauss_jordan(A,b=None):
    if type(A) is not np.ndarray:
        A = np.array(A)
        
    assert(np.linalg.det(A) != 0)
    
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
    

def lu(A,b):
    assert(np.linalg.det(A) != 0)
    
    L,U,_ = ref(A,b)
    d = forward_sub(L,b)
    x = back_sub(U,d)
    
    return x