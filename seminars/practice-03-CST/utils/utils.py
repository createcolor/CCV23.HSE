import pandas as pd
import numpy  as np
import scipy  as sp

RP_REGRESSION_TERMS_P1 = np.array([
    [1  ,0  ,0  ], [0  , 1  , 0  ], [0  , 0  , 1  ]
])
RP_REGRESSION_TERMS_P2 = np.concatenate((RP_REGRESSION_TERMS_P1, 
np.array([
    [1/2, 1/2, 0], [1/2, 0  , 1/2], [0  , 1/2, 1/2]
])))
RP_REGRESSION_TERMS_P3 = np.concatenate((RP_REGRESSION_TERMS_P2, 
np.array([
    [1/3, 2/3, 0  ], [0  , 1/3, 2/3], [1/3, 0, 2/3], 
    [2/3, 1/3, 0  ], [0  , 2/3, 1/3], [2/3, 0, 1/3], 
    [1/3, 1/3, 1/3]
])))
RP_REGRESSION_TERMS_P4 = np.concatenate((RP_REGRESSION_TERMS_P3, 
np.array([
    [3/4, 1/4, 0  ], [3/4, 0  , 1/4], [1/4, 3/4, 0  ], 
    [0  , 3/4, 1/4], [1/4, 0  , 3/4], [0, 1/4, 3/4  ], 
    [2/4, 1/4, 1/4], [1/4, 2/4, 1/4], [1/4, 1/4, 2/4] 
])))
RP_REGRESSION_TERMS_P5 = np.concatenate((RP_REGRESSION_TERMS_P4, 
np.array([
    [4/5, 1/5, 0  ], [4/5, 0  , 1/5], [3/5, 2/5, 0  ], 
    [3/5, 1/5, 1/5], [3/5, 0  , 2/5], [2/5, 3/5, 0  ], 
    [2/5, 2/5, 1/5], [2/5, 1/5, 2/5], [2/5, 0  , 3/5], 
    [1/5, 4/5, 0  ], [1/5, 3/5, 1/5], [1/5, 2/5, 2/5], 
    [1/5, 1/5, 3/5], [1/5, 0  , 4/5], [0  , 4/5, 1/5], 
    [0  , 3/5, 2/5], [0  , 2/5, 3/5], [0  , 1/5, 4/5]
])))
RP_REGRESSION_TERMS = np.array([RP_REGRESSION_TERMS_P1, RP_REGRESSION_TERMS_P2, RP_REGRESSION_TERMS_P3, RP_REGRESSION_TERMS_P4, RP_REGRESSION_TERMS_P5], dtype=object)

class RootPolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
        self.is_fitted_ = False
        if degree > 5: self.degree_ = 5
        if degree < 0: raise ValueError('degree must not be less than zero')
    
    def fit(self, X, y = None):
        self.n_output_features_ = 3
        self.terms_ = RP_REGRESSION_TERMS[0]
        if self.degree == 2: 
            self.n_output_features_ = 6
            self.terms_ = RP_REGRESSION_TERMS[1]
        if self.degree == 3:
            self.n_output_features_ = 13
            self.terms_ = RP_REGRESSION_TERMS[2]
        if self.degree == 4: 
            self.n_output_features_ = 22
            self.terms_ = RP_REGRESSION_TERMS[3]
        if self.degree == 5: 
            self.n_output_features_ = 40
            self.terms_ = RP_REGRESSION_TERMS[4]
        self.n_samples_, self.n_input_features_ = X.shape
        self.is_fitted_ = True
        
        return self

    def transform(self, X, y=None):
        if not self.is_fitted_:
            raise ValueError('Not fitted yet')
        features = []
        for term in self.terms_:
            features.append(np.prod(np.power(X, term), axis = 1))
        return np.array(features).T
    
    def fit_transform(self, X, y=None):
        self = self.fit(X, y)
        return self.transform(X, y)


FRAC_RAT_TERMS_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
FRAC_RAT_TERMS_2 = np.concatenate((FRAC_RAT_TERMS_1, np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])))
FRAC_RAT_TERMS_3 = np.concatenate((FRAC_RAT_TERMS_2, np.array([[1, 2, 0], [1, 1, 1], [1, 0, 2], [0, 1, 2]])))
FRAC_RAT_TERMS_4 = np.concatenate((FRAC_RAT_TERMS_3, np.array([[2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 0, 2], [0, 2, 2]])))
FRAC_RAT_TERMS_5 = np.concatenate((FRAC_RAT_TERMS_4, np.array([[2, 3, 0], [3, 0, 2], [0, 2, 3], [2, 2, 1], [1, 2, 2], [2, 1, 2]])))
FRAC_RAT_TERMS_6 = np.concatenate((FRAC_RAT_TERMS_5, np.array([[3, 3, 0], [3, 0, 3], [0, 3, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3], [2, 2, 2]])))

FRAC_RAT_TERMS = np.array([FRAC_RAT_TERMS_1, FRAC_RAT_TERMS_2, FRAC_RAT_TERMS_3, FRAC_RAT_TERMS_4, FRAC_RAT_TERMS_5, FRAC_RAT_TERMS_6], dtype=object)

class RobustScalableRationalFeatures:
    def __init__(self, degree, eps = 1.e-9):
        self.degree = degree
        self.eps    = eps
        self.is_fitted_ = False
        if degree > 6: self.degree = 6
        if degree < 0: raise ValueError('degree must not be less than zero')
    
    def fit(self, X, y = None):
        if self.degree == 1: 
            self.n_output_features_ = len(FRAC_RAT_TERMS_1)
            self.terms_ = FRAC_RAT_TERMS[0]
        if self.degree == 2: 
            self.n_output_features_ = len(FRAC_RAT_TERMS_2)
            self.terms_ = FRAC_RAT_TERMS[1]
        if self.degree == 3:
            self.n_output_features_ = len(FRAC_RAT_TERMS_3)
            self.terms_ = FRAC_RAT_TERMS[2]
        if self.degree == 4: 
            self.n_output_features_ = len(FRAC_RAT_TERMS_4)
            self.terms_ = FRAC_RAT_TERMS[3]
        if self.degree == 5: 
            self.n_output_features_ = len(FRAC_RAT_TERMS_5)
            self.terms_ = FRAC_RAT_TERMS[4]
        self.n_samples_, self.n_input_features_ = X.shape
        self.is_fitted_ = True
        
        return self
    def transform(self, X, y=None):
        if not self.is_fitted_:
            raise ValueError('Not fitted yet')
        features = []
        for term in self.terms_:
            features.append(np.prod(np.power(X, term), axis = 1))
            
        features = np.array(features).T
        
        sum_features = np.sum(X, axis=1, keepdims=True) + self.eps
        
        if self.degree >= 2:
            lft_index = FRAC_RAT_TERMS_1.shape[0]
            rht_index = FRAC_RAT_TERMS_2.shape[0]
            features[:,lft_index: rht_index] /= (sum_features)**1
        if self.degree >= 3:
            lft_index = FRAC_RAT_TERMS_2.shape[0]
            rht_index = FRAC_RAT_TERMS_3.shape[0]
            features[:,lft_index:rht_index] /= (sum_features)**2
        if self.degree >= 4:
            lft_index = FRAC_RAT_TERMS_3.shape[0]
            rht_index = FRAC_RAT_TERMS_4.shape[0]
            features[:,lft_index:rht_index] /= (sum_features)**3
        if self.degree >= 5:
            lft_index = FRAC_RAT_TERMS_4.shape[0]
            rht_index = FRAC_RAT_TERMS_5.shape[0]
            features[:,lft_index:rht_index] /= (sum_features)**4
                        
        return features
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

def read_h5(h5_path: str, wv_hsi: np.ndarray):
    with h5py.File(h5_path, 'r') as fh:
        hsi_y = np.transpose(fh['img\\']).astype(np.float32)
        hsi = SpectralFunction(hsi_y, wv_hsi, dtype=np.float32)
        return hsi


