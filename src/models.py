import numpy as np

# REGRESSION MODELS
class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit Linear Regression model using Normal Equation
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        
        Returns:
        --------
        self : object
        """
        if self.fit_intercept:
            # Add ones column for intercept
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        else:
            X_with_intercept = X
        
        # Normal Equation: β = (X^T X)^(-1) X^T y
        # Use solve instead of inv to avoid numerical issues
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        weights = np.linalg.solve(XtX, Xty)
        
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
        """
        return X @ self.coef_ + self.intercept_

# LASSO 
class Lasso:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
    
    def _soft_threshold(self, rho, alpha):
        """
        Vectorized soft-thresholding operator
        S(rho, alpha) = sign(rho) * max(|rho| - alpha, 0)
        """
        return np.sign(rho) * np.maximum(np.abs(rho) - alpha, 0)
    
    def fit(self, X, y):
        """
        Fit Lasso model using Coordinate Descent
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        
        Returns:
        --------
        self : object
        """
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X_with_intercept = np.c_[np.ones(n_samples), X]
        else:
            X_with_intercept = X
        
        # Initialize weights = 0
        weights = np.zeros(X_with_intercept.shape[1])
        
        # Pre-compute X^T @ X diagonal for optimization
        XtX_diag = np.sum(X_with_intercept ** 2, axis=0)
        
        # Coordinate Descent
        for iteration in range(self.max_iter):
            weights_old = weights.copy()
            
            # Vectorized update for all weights
            for j in range(len(weights)):
                # Calculate residual excluding feature j
                residual = y - X_with_intercept @ weights + weights[j] * X_with_intercept[:, j]
                
                # Calculate rho_j = X_j^T * residual
                rho = X_with_intercept[:, j] @ residual
                
                # Update weight j
                if j == 0 and self.fit_intercept:
                    # Don't regularize intercept
                    weights[j] = rho / n_samples
                else:
                    # Soft-thresholding with alpha
                    weights[j] = self._soft_threshold(rho / n_samples, self.alpha) / (XtX_diag[j] / n_samples)
            
            # Check convergence
            if np.sum(np.abs(weights - weights_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        
        return self
    
    def predict(self, X):
        """
        Predict using the Lasso model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
        """
        return X @ self.coef_ + self.intercept_



# CROSS-VALIDATION
class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """
        Generate indices to split data into training and test set
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,), optional
        
        Yields:
        -------
        train_idx : ndarray
            The training set indices for that split
        test_idx : ndarray
            The testing set indices for that split
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        # Vectorized split calculation
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop


# METRICS
def r2_score(y_true, y_pred):
    """
    Calculate R² score (Coefficient of Determination)
    
    R² = 1 - (SS_res / SS_tot)
    SS_res = sum of squared residuals
    SS_tot = total sum of squares
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        Ground truth target values
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns:
    --------
    score : float
        R² score, best value is 1.0
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot)


def rmse_score(y_true, y_pred):
    """
    Calculate RMSE (Root Mean Squared Error)
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        Ground truth target values
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns:
    --------
    score : float
        RMSE score, best value is 0.0
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae_score(y_true, y_pred):
    """
    Calculate MAE (Mean Absolute Error)
    
    MAE = mean(|y_true - y_pred|)
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        Ground truth target values
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns:
    --------
    score : float
        MAE score, best value is 0.0
    """
    return np.mean(np.abs(y_true - y_pred))


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model with multiple metrics
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        Ground truth target values
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    model_name : str, default="Model"
        Name of the model for display
    
    Returns:
    --------
    metrics : dict
        Dictionary containing R², RMSE, and MAE scores
    """
    r2 = r2_score(y_true, y_pred)
    rmse = rmse_score(y_true, y_pred)
    mae = mae_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def cross_validate(estimator, X, y, cv=5):
    """
    Evaluate estimator by cross-validation
    
    Parameters:
    -----------
    estimator : estimator object
        Object with fit and predict methods
    X : array-like, features
    y : array-like, target
    cv : int or cross-validation generator
        If int, KFold with cv splits will be used
    
    Returns:
    --------
    scores : dict
        Dictionary containing R², RMSE, and MAE scores for each fold
    """
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    print(f"\nPerforming {cv.n_splits if hasattr(cv, 'n_splits') else 'K'}-Fold Cross-Validation...")
    print(f"Model: {type(estimator).__name__}" + 
          (f" (alpha={estimator.alpha})" if hasattr(estimator, 'alpha') else ""))
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data - vectorized indexing
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Clone estimator and fit
        if hasattr(estimator, 'alpha'):
            fold_estimator = type(estimator)(alpha=estimator.alpha, 
                                             max_iter=estimator.max_iter, 
                                             tol=estimator.tol)
        else:
            fold_estimator = type(estimator)()
        
        fold_estimator.fit(X_fold_train, y_fold_train)
        
        # Predict on validation set
        y_fold_pred = fold_estimator.predict(X_fold_val)
        
        # Calculate metrics - vectorized
        r2 = r2_score(y_fold_val, y_fold_pred)
        rmse = rmse_score(y_fold_val, y_fold_pred)
        mae = mae_score(y_fold_val, y_fold_pred)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"  Fold {fold_idx + 1}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    
    # Vectorized mean and std calculation
    r2_scores = np.array(r2_scores)
    rmse_scores = np.array(rmse_scores)
    mae_scores = np.array(mae_scores)
    
    print(f"\n  Mean R²: {r2_scores.mean():.3f} (+/- {r2_scores.std():.4f})")
    print(f"  Mean RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std():.4f})")
    print(f"  Mean MAE: {mae_scores.mean():.3f} (+/- {mae_scores.std():.4f})")
    
    return {
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'mean_r2': r2_scores.mean(),
        'std_r2': r2_scores.std(),
        'mean_rmse': rmse_scores.mean(),
        'std_rmse': rmse_scores.std(),
        'mean_mae': mae_scores.mean(),
        'std_mae': mae_scores.std()
    }
