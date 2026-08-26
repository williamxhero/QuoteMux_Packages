"""Administrative archive builder for the authorized S000012 Pyramid artifact.

It deliberately registers no QuoteMux runtime capability.  The resulting bundle is
an immutable import candidate whose lineage remains user-provided and unverified.
"""

from .bundle import build_bundle, preflight, validate_authorization

__all__ = ["build_bundle", "preflight", "validate_authorization"]
