# Compatibility shim for older imports
# Exposes FeatureExtractionPipeline and FeatureExtractionManager
from extract_features_90plus import FeatureExtractionPipeline

# Backwards-compatible alias
FeatureExtractionManager = FeatureExtractionPipeline

__all__ = ['FeatureExtractionPipeline', 'FeatureExtractionManager']
