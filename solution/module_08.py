"""Analytic covariance closure for architectures outside the fitted model."""
import flopscope as fs
import flopscope.numpy as f


def general_response(mlp):
    n = mlp.width
    mu = f.zeros(n, dtype=f.float64)
    covariance = f.eye(n, dtype=f.float64)
    rows = []
    for weight in mlp.weights:
        w = f.asarray(weight, dtype=f.float64)
        pre_mu = w.T @ mu
        pre_cov = w.T @ (covariance @ w)
        variance = f.maximum(f.diag(pre_cov), 0.)
        sigma = f.sqrt(variance)
        alpha = pre_mu / f.maximum(sigma, 1e-150)
        gate = fs.stats.norm.cdf(alpha)
        density = fs.stats.norm.pdf(alpha)
        mu = f.maximum(pre_mu * gate + sigma * density, 0.)
        second = (pre_mu * pre_mu + variance) * gate + pre_mu * sigma * density
        inherited = variance * gate * gate
        # Var(ReLU(X)) >= Var(X) * E[ReLU'(X)]**2 for Gaussian X.
        # Enforcing this exact inequality prevents cancellation from producing
        # a negative diagonal increment in the positive covariance closure.
        post_variance = f.maximum(second - mu * mu, inherited)
        covariance = pre_cov * (gate[:, None] * gate[None, :])
        covariance = (covariance + covariance.T) * 0.5
        f.fill_diagonal(covariance, post_variance)
        rows.append(mu)
    return f.stack(rows, axis=0)
