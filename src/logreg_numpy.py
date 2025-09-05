import numpy as np
class LogisticRegressionNumpy:
def init(self, lr=0.1, n_iters=2000, l2=0.0, fit_intercept=True, random_state=42):
self.lr = lr
self.n_iters = n_iters
self.l2 = l2
self.fit_intercept = fit_intercept
self.rng = np.random.default_rng(random_state)
self.w = None
@staticmethod
def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1 + ez)
    return out

def _add_intercept(self, X):
    if not self.fit_intercept:
        return X
    return np.hstack([np.ones((X.shape[0], 1)), X])

def fit(self, X, y):
    X = self._add_intercept(X)
    n, d = X.shape
    self.w = self.rng.normal(scale=0.01, size=d)
    for _ in range(self.n_iters):
        z = X @ self.w
        p = self._sigmoid(z)
        err = p - y
        grad = (X.T @ err) / n
        if self.l2 > 0:
            reg = self.w.copy()
            if self.fit_intercept:
                reg[0] = 0.0
            grad += self.l2 * reg / n
        self.w -= self.lr * grad
    return self

def predict_proba(self, X):
    X = self._add_intercept(X)
    return self._sigmoid(X @ self.w)

def predict(self, X, threshold=0.5):
    return (self.predict_proba(X) >= threshold).astype(int)

