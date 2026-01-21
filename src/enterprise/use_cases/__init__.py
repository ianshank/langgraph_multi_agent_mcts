"""
Enterprise Use Case Implementations.

This package contains concrete implementations of enterprise use cases:
- M&A Due Diligence
- Clinical Trial Design
- Regulatory Compliance
"""

from __future__ import annotations


# Lazy imports to avoid circular dependencies
def get_ma_due_diligence():
    from .ma_due_diligence import MADueDiligence

    return MADueDiligence


def get_clinical_trial():
    from .clinical_trial import ClinicalTrialDesign

    return ClinicalTrialDesign


def get_regulatory_compliance():
    from .regulatory_compliance import RegulatoryCompliance

    return RegulatoryCompliance


__all__ = [
    "get_ma_due_diligence",
    "get_clinical_trial",
    "get_regulatory_compliance",
]
