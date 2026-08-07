"""
image_model — Image Authenticity Verification package.

Classes: Real Photograph vs AI Generated (configurable)

Provides ImageAuthenticityVerifier for pre-validation of disaster imagery.
"""

from image_model.inference.predictor import (
    ImageAuthenticityVerifier,
    AuthenticityVerificationResult,
    ImagePredictor,
)

__all__ = [
    "ImageAuthenticityVerifier",
    "AuthenticityVerificationResult",
    "ImagePredictor",
]
