import type { TranslationKey } from "./i18n";

const warningKeys: Record<string, TranslationKey> = {
  "provider.global_prediction.model_not_installed": "warnings.modelNotInstalled",
  "provider.global_prediction.inference_timeout": "warnings.modelTimeout",
  "provider.global_prediction.invalid_model_output": "warnings.modelInvalidOutput",
  "provider.global_prediction.provider_initialization_failed": "warnings.modelProviderFailed",
  "provider.global_prediction.cuda_out_of_memory": "warnings.modelProviderFailed",
  "provider.global_prediction.weights_incomplete": "warnings.modelIncomplete",
  "provider.global_prediction.checksum_mismatch": "warnings.modelIntegrity",
  "provider.global_prediction.unsupported_device": "warnings.modelDevice",
  "provider.global_prediction.missing_dependency": "warnings.modelDependency",
  "provider.ocr.timeout": "warnings.ocrTimeout",
  "provider.ocr.unavailable": "warnings.ocrUnavailable",
  "phase5b.ocr.disabled": "warnings.ocrUnavailable",
  "provider.segmentation.timeout": "warnings.segmentationTimeout",
  "provider.segmentation.unavailable": "warnings.segmentationUnavailable",
  "provider.segmentation.model_not_installed": "warnings.segmentationUnavailable",
  "provider.segmentation.invalid_output": "warnings.segmentationInvalidOutput",
  "provider.segmentation.inference_failed": "warnings.segmentationUnavailable",
  "provider.segmentation.failed": "warnings.segmentationUnavailable",
  "provider.segmentation.missing_dependency": "warnings.segmentationUnavailable",
  "provider.scene_segmentation.timeout": "warnings.segmentationTimeout",
  "provider.scene_segmentation.unavailable": "warnings.segmentationUnavailable",
  "provider.scene_segmentation.invalid_output": "warnings.segmentationInvalidOutput",
  "provider.scene_segmentation.failed": "warnings.segmentationUnavailable",
  "provider.reverse_geocoding.timeout": "warnings.reverseGeocodingTimeout",
  "provider.reverse_geocoding.unavailable": "warnings.reverseGeocodingUnavailable",
  "provider.reverse_geocoding.failed": "warnings.reverseGeocodingUnavailable",
  "provider.reverse_geocoding.cache_unavailable": "warnings.reverseGeocodingCache",
  "provider.geocoding.timeout": "warnings.reverseGeocodingTimeout",
  "provider.geocoding.unavailable": "warnings.reverseGeocodingUnavailable",
  "warning.global_prediction_uncalibrated": "warnings.globalUncalibrated",
  "warning.confidence_not_calibrated": "warnings.confidenceUncalibrated",
};

export function providerWarningKey(warning: string): TranslationKey {
  return warningKeys[warning] ?? "warnings.additional";
}
