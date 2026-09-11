"""
Feature extraction and classical-model pipeline builders. 
"""
import numpy as np
from scipy.signal import butter, filtfilt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM


def extract_band_power(X, sfreq=160., channel_idx=None):
    """Mean-variance band power in mu (8-12Hz) and beta (13-30Hz), log-transformed
    since power is heavily right-skewed and LDA assumes roughly Gaussian
    class-conditional features (the same reason CSP always logs its output).
    channel_idx: optional list of channel indices to restrict to (e.g. motor
    strip only) — if None, uses all channels."""
    def bandpower(data, lo, hi, fs):
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
        power = filtfilt(b, a, data, axis=-1).var(axis=-1)
        return np.log(power + 1e-10)

    if channel_idx is not None:
        X = X[:, channel_idx, :]

    return np.concatenate(
        [bandpower(X, 8, 12, sfreq), bandpower(X, 13, 30, sfreq)], axis=1
    )


def riemannian_align(X, ref_cov=None):
    """Recenters covariance matrices toward a common reference before
    classification. Rescales each trial's covariance by the subject's own
    mean covariance, pulling everyone's data toward a shared point on the
    manifold — directly targets the individual differences in baseline
    covariance structure that have nothing to do with the task itself.
    Unsupervised: uses only the trials' own covariances, no labels, so it's
    legitimate to apply per-subject including the held-out LOSO test subject
    without leakage."""
    from pyriemann.utils.mean import mean_riemann
    from scipy.linalg import fractional_matrix_power

    cov = Covariances(estimator='lwf').fit_transform(X)
    if ref_cov is None:
        ref_cov = mean_riemann(cov)
    ref_inv_sqrt = fractional_matrix_power(ref_cov, -0.5)
    aligned = np.array([ref_inv_sqrt @ c @ ref_inv_sqrt.T for c in cov])
    return aligned, ref_cov


def build_csp_pipeline():
    """CSP spatial filters + log-variance features + shrinkage LDA."""
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
    return make_pipeline(csp, StandardScaler(), LDA(solver='eigen', shrinkage='auto'))


def build_mdm_pipeline(pre_aligned=False):
    """Minimum Distance to Mean (Barachant et al. 2013) — classifies directly
    on the manifold via geodesic distance to each class's Riemannian mean
    covariance, no tangent-space vectorization and no parametric classifier
    fit on top. Structurally resistant to the overfitting seen in earlier
    tangent-space + logistic regression attempts (too many features, too few
    trials per fold, see notebook): confirmed via learning-curve diagnostic
    that train and held-out accuracy stay close together across all training
    set sizes here, unlike those variants.

    pre_aligned: set True when the input has already passed through
    riemannian_align() (e.g. in 04_riemannian_mdm_aligned.py) — alignment
    already produces covariance matrices, so the Covariances step is skipped."""
    if pre_aligned:
        return make_pipeline(MDM(metric='riemann'))
    return make_pipeline(Covariances(estimator='lwf'), MDM(metric='riemann'))


