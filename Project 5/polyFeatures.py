import numpy as np

def polyFeatures(X, p):
    """
        Функция позволяет создать дополнительные свойства для построения  
        модели на основе полиномиальной регрессия, позволяя на основе 
        признака X[i, :] создать вектор-признаков X_poly[i, :] = 
        [X[i, :] X[i, :].^2 X[i, :].^3 ...  X[i, :].^p]
    """
    
    X_poly = np.zeros([X.shape[0], p])
    
    # ====================== Ваш код здесь ======================
    # Инструкция: вычислить дополнительные свойства для построения  
    # модели на основе полиномиальной регрессия
    
    
    
    # ============================================================
    
    return X_poly