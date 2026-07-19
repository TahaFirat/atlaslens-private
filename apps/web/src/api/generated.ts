// Generated from packages/contracts/openapi.yaml. Do not edit manually.
export interface paths {
    "/api/v1/health": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Liveness probe */
        get: operations["getHealth"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/ready": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Required local dependency readiness */
        get: operations["getReadiness"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/capabilities": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Discover safe runtime capabilities */
        get: operations["getCapabilities"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/providers": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List safe provider and deployment status */
        get: operations["listProviders"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/models": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List safe registered model status */
        get: operations["listModels"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/analyses": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List privacy-bounded analysis history */
        get: operations["listAnalyses"];
        put?: never;
        /** Create an asynchronous image analysis */
        post: operations["createAnalysis"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/analyses/{analysis_id}/rerun": {
        parameters: {
            query?: never;
            header?: never;
            path: {
                analysis_id: components["parameters"]["AnalysisId"];
            };
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Rerun only when the original source was explicitly retained */
        post: operations["rerunAnalysis"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/analyses/{analysis_id}": {
        parameters: {
            query?: never;
            header?: never;
            path: {
                analysis_id: components["parameters"]["AnalysisId"];
            };
            cookie?: never;
        };
        /** Get current or terminal analysis state */
        get: operations["getAnalysis"];
        put?: never;
        post?: never;
        /** Delete or cancel an analysis */
        delete: operations["deleteAnalysis"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/analyses/{analysis_id}/events": {
        parameters: {
            query?: never;
            header?: never;
            path: {
                analysis_id: components["parameters"]["AnalysisId"];
            };
            cookie?: never;
        };
        /** Stream progress, heartbeat, and terminal events */
        get: operations["streamAnalysisEvents"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/evaluations": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List validated benchmark report summaries */
        get: operations["listEvaluations"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/evaluations/{report_id}": {
        parameters: {
            query?: never;
            header?: never;
            path: {
                report_id: string;
            };
            cookie?: never;
        };
        /** Get one validated benchmark report summary */
        get: operations["getEvaluation"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/datasets/qa": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List prebuilt safe dataset-QA reports */
        get: operations["listDatasetQaReports"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/datasets/qa/{report_id}": {
        parameters: {
            query?: never;
            header?: never;
            path: {
                report_id: string;
            };
            cookie?: never;
        };
        /** Get one prebuilt safe dataset-QA report */
        get: operations["getDatasetQaReport"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/system-intelligence": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Safe model and reference-index runtime inventory */
        get: operations["getSystemIntelligence"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/mapillary-demo/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get private Mapillary demo readiness */
        get: operations["getMapillaryDemoStatus"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/mapillary-demo/query": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Query the private attributed Mapillary demo */
        post: operations["queryMapillaryDemo"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Cases */
        get: operations["listCases"];
        put?: never;
        /**
         * Create Case
         * @description Create a case in the local default workspace seam; this does not claim production tenant isolation.
         */
        post: operations["createCase"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Case */
        get: operations["getCase"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /** Patch Case */
        patch: operations["patchCase"];
        trace?: never;
    };
    "/api/v1/cases/{case_id}/media": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** List Case Media */
        get: operations["listCaseMedia"];
        put?: never;
        /**
         * Create Case Media
         * @description Store sanitized image metadata only. Image bytes must use the existing POST /api/v1/analyses workflow.
         */
        post: operations["createCaseMedia"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/analyses/{analysis_id}/link": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Link Analysis
         * @description Idempotently link one existing analysis to explicitly identified case media after fail-closed relationship checks.
         */
        post: operations["linkCaseAnalysis"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/analyses/{analysis_id}/materialize-evidence": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Materialize Evidence
         * @description Idempotently normalize retained analysis metadata into immutable evidence and hypotheses; no image pixels are stored here.
         */
        post: operations["materializeCaseEvidence"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/evidence": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Evidence
         * @description List immutable normalized evidence in stable creation order. Retrieval evidence is not a confirmed location.
         */
        get: operations["listCaseEvidence"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/hypotheses": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Hypotheses
         * @description List model, imported, and separate operator hypotheses without manufacturing probability values.
         */
        get: operations["listCaseHypotheses"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/hypotheses/{hypothesis_id}/adjudications": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Add Adjudication
         * @description Append an analyst decision; prior adjudications are never updated or deleted.
         */
        post: operations["createHypothesisAdjudication"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/operator-hypotheses": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Add Operator Hypothesis
         * @description Create a separate operator-origin hypothesis and initial adjudication without overwriting model output.
         */
        post: operations["createOperatorHypothesis"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/audit-events": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Audit Events
         * @description List append-only, hash-chained application history in monotonic sequence order.
         */
        get: operations["listCaseAuditEvents"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/cases/{case_id}/audit-integrity": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Audit Integrity
         * @description Verify tamper-evident application history. This is not legally certified evidence.
         */
        get: operations["getCaseAuditIntegrity"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /**
         * AnalysisMode
         * @enum {string}
         */
        AnalysisMode: "local_only" | "cloud_assisted";
        /**
         * AnalysisStatus
         * @enum {string}
         */
        AnalysisStatus: "queued" | "processing" | "completed" | "failed" | "deleted";
        /** HealthResponse */
        HealthResponse: {
            /**
             * Status
             * @default ok
             * @constant
             */
            status: "ok";
            /** Version */
            version: string;
        };
        /** ReadinessResponse */
        ReadinessResponse: {
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "not_ready";
            /** Checks */
            checks: {
                [key: string]: "ok" | "error";
            };
        };
        /** ProviderCapability */
        ProviderCapability: {
            /** Provider Id */
            provider_id: string;
            /** Enabled */
            enabled: boolean;
            /** Available */
            available: boolean;
            /**
             * Execution Boundary
             * @enum {string}
             */
            execution_boundary: "local" | "cloud";
            /** Reason Code */
            reason_code?: string | null;
            /** Installed */
            installed?: boolean | null;
            /** Verified */
            verified?: boolean | null;
            /** Operational Status */
            operational_status?: ("ready" | "not_ready" | "not_installed" | "dependencies_installed" | "weights_prepared" | "worker_unreachable" | "model_load_failed" | "inference_not_verified" | "incomplete" | "loading" | "failed" | "disabled" | "unavailable") | null;
            /** Model Name */
            model_name?: string | null;
            /** Model Revision */
            model_revision?: string | null;
            /** Device */
            device?: string | null;
            /** Calibration State */
            calibration_state?: ("uncalibrated" | "preliminary" | "calibrated") | null;
            /** Provider Type */
            provider_type?: string | null;
            /** Offline */
            offline?: boolean | null;
            /** License */
            license?: string | null;
            /** Limitation */
            limitation?: string | null;
            /** Weights Available */
            weights_available?: boolean | null;
            /** Usable */
            usable?: boolean | null;
            /** Execution Mode */
            execution_mode?: ("in_process" | "isolated_worker") | null;
            /** Source Revision */
            source_revision?: string | null;
            /** Load Error */
            load_error?: string | null;
            /** Key Configured */
            key_configured?: boolean | null;
            /** Budget Available */
            budget_available?: boolean | null;
        };
        /** RetentionPolicy */
        RetentionPolicy: {
            /** Keep Uploads */
            keep_uploads: boolean;
            /** Ttl Seconds */
            ttl_seconds: number;
            /** Originals Deleted After Analysis */
            originals_deleted_after_analysis: boolean;
        };
        /** CapabilitiesResponse */
        CapabilitiesResponse: {
            /** Supported Formats */
            supported_formats: ("jpeg" | "png" | "webp")[];
            /** Max Upload Bytes */
            max_upload_bytes: number;
            /** Max Decoded Pixels */
            max_decoded_pixels: number;
            /** Enabled Analysis Modes */
            enabled_analysis_modes: components["schemas"]["AnalysisMode"][];
            /** Providers */
            providers: {
                [key: string]: components["schemas"]["ProviderCapability"];
            };
            retention: components["schemas"]["RetentionPolicy"];
            /** Version */
            version: string;
        };
        /** ProviderStatusItem */
        ProviderStatusItem: {
            /** Provider Id */
            provider_id: string;
            /** Provider Type */
            provider_type: string;
            /**
             * Mode
             * @enum {string}
             */
            mode: "disabled" | "shadow" | "candidate" | "primary";
            /** Available */
            available: boolean;
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "not_ready" | "not_installed" | "dependencies_installed" | "weights_prepared" | "worker_unreachable" | "model_load_failed" | "inference_not_verified" | "incomplete" | "loading" | "failed" | "disabled" | "unavailable";
            /**
             * Classification
             * @enum {string}
             */
            classification: "real" | "simulated";
            /** Model Name */
            model_name?: string | null;
            /** Model Revision */
            model_revision?: string | null;
            /** Device */
            device?: string | null;
            /** Calibration State */
            calibration_state?: ("uncalibrated" | "preliminary" | "calibrated") | null;
            /** Reason Code */
            reason_code?: string | null;
            /** Weights Available */
            weights_available?: boolean | null;
            /** Usable */
            usable?: boolean | null;
            /** Execution Mode */
            execution_mode?: ("in_process" | "isolated_worker") | null;
            /** Source Revision */
            source_revision?: string | null;
            /** Load Error */
            load_error?: string | null;
            /** Key Configured */
            key_configured?: boolean | null;
            /** Budget Available */
            budget_available?: boolean | null;
        };
        /** ProviderStatusResponse */
        ProviderStatusResponse: {
            /** Providers */
            providers: components["schemas"]["ProviderStatusItem"][];
        };
        /** ModelStatusItem */
        ModelStatusItem: {
            /** Model Id */
            model_id: string;
            /** Model Version */
            model_version?: string | null;
            /** Artifact Digest */
            artifact_digest?: string | null;
            /**
             * Mode
             * @enum {string}
             */
            mode: "disabled" | "shadow" | "candidate" | "primary";
            /** Verified */
            verified: boolean;
            /**
             * Status
             * @enum {string}
             */
            status: "not_registered" | "registered" | "verified" | "invalid" | "disabled";
            /** Reason Code */
            reason_code?: string | null;
        };
        /** ModelStatusResponse */
        ModelStatusResponse: {
            /** Models */
            models: components["schemas"]["ModelStatusItem"][];
        };
        /** AnalysisAccepted */
        AnalysisAccepted: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Status
             * @default queued
             * @constant
             */
            status: "queued";
            /** Status Url */
            status_url: string;
            /** Events Url */
            events_url: string;
            /** Delete Url */
            delete_url: string;
        };
        /** Progress */
        Progress: {
            /** Stage */
            stage: string;
            /** Percent */
            percent: number;
            /** Message Key */
            message_key: string;
        };
        /** ImageSummary */
        ImageSummary: {
            /**
             * Format
             * @enum {string}
             */
            format: "jpeg" | "png" | "webp";
            /** Width */
            width: number;
            /** Height */
            height: number;
            /** Megapixels */
            megapixels: number;
            /** Sha256 */
            sha256: string;
            /** Orientation Normalized */
            orientation_normalized: boolean;
            /** Exif Present */
            exif_present: boolean;
        };
        /** QualitySummary */
        QualitySummary: {
            /** Blur Score */
            blur_score: number;
            /** Brightness Score */
            brightness_score: number;
            /** Contrast Score */
            contrast_score: number;
            /** Resolution Score */
            resolution_score: number;
            /** Warnings */
            warnings: string[];
        };
        /** Provenance */
        Provenance: {
            /** Provider Id */
            provider_id: string;
            /** Provider Kind */
            provider_kind: string;
            /** Provider Version */
            provider_version: string;
            /**
             * Execution Boundary
             * @enum {string}
             */
            execution_boundary: "local" | "cloud";
            /** Model Name */
            model_name?: string | null;
            /** Output Schema Version */
            output_schema_version: string;
        };
        /**
         * RetrievalCoverageContext
         * @description Coverage and score semantics carried without inference-time context.
         */
        RetrievalCoverageContext: {
            /**
             * Analysis Scope
             * @enum {string}
             */
            analysis_scope: "generic_upload" | "ankara_reference_pilot" | "coarse_provider_routed";
            /**
             * Coverage Status
             * @enum {string}
             */
            coverage_status: "pilot_eligible" | "insufficient" | "provider_unavailable";
            /** Coverage Label */
            coverage_label: string;
            /** Retrieval Provider */
            retrieval_provider: string;
            /** Retrieval Scope */
            retrieval_scope: string;
            /** Result Semantics */
            result_semantics: string;
            /** Abstained */
            abstained: boolean;
            /** Abstention Reason */
            abstention_reason?: string | null;
            /**
             * Similarity Semantics
             * @default cosine_similarity_not_confidence
             * @constant
             */
            similarity_semantics: "cosine_similarity_not_confidence";
            /** Supported Region */
            supported_region: string;
            /** Evidence Version */
            evidence_version: string;
            /** Benchmark Version */
            benchmark_version: string;
        };
        /** Evidence */
        Evidence: {
            /** Id */
            id: string;
            /** Type */
            type: string;
            /** Label */
            label: string;
            /** Display Value */
            display_value: string | null;
            /** Confidence */
            confidence: number | null;
            /** Confidence Basis */
            confidence_basis: string;
            /** Source */
            source: string;
            /** Sensitive */
            sensitive: boolean;
            provenance: components["schemas"]["Provenance"];
            retrieval_context?: components["schemas"]["RetrievalCoverageContext"] | null;
        };
        /** GeoPoint */
        GeoPoint: {
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
        };
        /** GeoJsonPoint */
        GeoJsonPoint: {
            /**
             * Type
             * @default Point
             * @constant
             */
            type: "Point";
            /** Coordinates */
            coordinates: [
                number,
                number
            ];
        };
        /** GeoJsonPolygon */
        GeoJsonPolygon: {
            /**
             * Type
             * @default Polygon
             * @constant
             */
            type: "Polygon";
            /** Coordinates */
            coordinates: [
                number,
                number
            ][][];
        };
        /** Candidate */
        Candidate: {
            /** Id */
            id: string;
            /** Rank */
            rank: number;
            center: components["schemas"]["GeoPoint"];
            /** Geometry */
            geometry: components["schemas"]["GeoJsonPoint"] | components["schemas"]["GeoJsonPolygon"];
            /** Radius Km */
            radius_km: number;
            /** Uncertainty Basis */
            uncertainty_basis: string;
            /** Confidence */
            confidence: number | null;
            /**
             * Confidence Kind
             * @enum {string}
             */
            confidence_kind: "source_reliability" | "uncalibrated_score" | "calibrated_probability";
            /** Confidence Basis */
            confidence_basis: string;
            /**
             * Granularity
             * @enum {string}
             */
            granularity: "exact_metadata" | "city" | "region" | "country" | "broad_area";
            /** Country Code */
            country_code?: string | null;
            /** Label */
            label?: string | null;
            /** Source */
            source: string;
            /** Evidence Ids */
            evidence_ids: string[];
            /** Evidence Summary */
            evidence_summary: string;
            /** Provenance */
            provenance: components["schemas"]["Provenance"][];
            /**
             * Verification Status
             * @enum {string}
             */
            verification_status: "metadata_only" | "unverified_model" | "corroborated" | "geometrically_verified";
            /** Verified */
            verified: boolean;
            phase4_assessment?: components["schemas"]["Phase4Assessment"] | null;
            phase5b_assessment?: components["schemas"]["Phase5BAssessment"] | null;
            model_prediction?: components["schemas"]["ModelPredictionDiagnostics"] | null;
            geoclip_cluster?: components["schemas"]["GeoClipClusterSummary"] | null;
            confidence_assessment?: components["schemas"]["UncalibratedConfidenceSummary"] | null;
            reverse_geocode?: components["schemas"]["ReverseGeocodeSummary"] | null;
        };
        /** SceneClassSummary */
        SceneClassSummary: {
            /** Class Id */
            class_id: number;
            /** Class Name */
            class_name: string;
            /** Pixel Ratio */
            pixel_ratio: number;
            /** Percentage */
            percentage: number;
        };
        /** SceneGroupSummary */
        SceneGroupSummary: {
            /** Name */
            name: string;
            /** Pixel Ratio */
            pixel_ratio: number;
            /** Percentage */
            percentage: number;
        };
        /** SceneTagSummary */
        SceneTagSummary: {
            /** Name */
            name: string;
            /** Strength */
            strength: number;
            /**
             * Strength Semantics
             * @default deterministic_heuristic_not_probability
             * @constant
             */
            strength_semantics: "deterministic_heuristic_not_probability";
            /** Reason */
            reason: string;
        };
        /** SceneSegmentationSummary */
        SceneSegmentationSummary: {
            /**
             * Status
             * @default completed
             * @constant
             */
            status: "completed";
            /** Provider */
            provider: string;
            /** Device */
            device: string;
            /** Inference Ms */
            inference_ms: number;
            /** Image Width */
            image_width: number;
            /** Image Height */
            image_height: number;
            /** Semantic Label Names Available */
            semantic_label_names_available: boolean;
            /** Dominant Classes */
            dominant_classes: components["schemas"]["SceneClassSummary"][];
            /** Scene Groups */
            scene_groups?: components["schemas"]["SceneGroupSummary"][];
            /** Scene Tags */
            scene_tags?: components["schemas"]["SceneTagSummary"][];
            /** Warnings */
            warnings?: string[];
        };
        /** Phase6BGeographicCandidateSummary */
        Phase6BGeographicCandidateSummary: {
            /** Candidate Id */
            candidate_id: string;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Raw Score */
            raw_score?: number | null;
            /** Provider Rank */
            provider_rank: number;
            /** Sample Support */
            sample_support: number;
            /** Metadata */
            metadata?: {
                [key: string]: string | number | boolean | null;
            };
        };
        /** Phase6BProviderPredictionSummary */
        Phase6BProviderPredictionSummary: {
            /** Provider */
            provider: string;
            /** Model Id */
            model_id: string;
            /** Model Revision */
            model_revision: string;
            /**
             * Source Family
             * @enum {string}
             */
            source_family: "mp16_family" | "osv5m_family" | "yfcc_family" | "inat_family" | "textual_evidence_family" | "cloud_reasoning_family";
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "skipped" | "disabled" | "failed" | "timeout";
            /**
             * Device
             * @enum {string}
             */
            device: "cuda" | "cpu";
            /** Duration Ms */
            duration_ms: number;
            /**
             * Score Semantics
             * @enum {string}
             */
            score_semantics: "similarity" | "direct_regression" | "sample_density";
            /** Candidates */
            candidates?: components["schemas"]["Phase6BGeographicCandidateSummary"][];
            /** Warnings */
            warnings?: string[];
            /** Reason Code */
            reason_code?: string | null;
            /** Diagnostics */
            diagnostics?: {
                [key: string]: string | number | boolean | null;
            };
        };
        /** Phase6BFusionMemberSummary */
        Phase6BFusionMemberSummary: {
            /** Provider */
            provider: string;
            /** Model Id */
            model_id: string;
            /** Candidate Id */
            candidate_id: string;
            /**
             * Source Family
             * @enum {string}
             */
            source_family: "mp16_family" | "osv5m_family" | "yfcc_family" | "inat_family" | "textual_evidence_family" | "cloud_reasoning_family";
            /** Provider Rank */
            provider_rank: number;
            /** Sample Support */
            sample_support: number;
        };
        /** Phase6BFusionContributionSummary */
        Phase6BFusionContributionSummary: {
            /** Name */
            name: string;
            /** Raw Value */
            raw_value: number;
            /** Weight */
            weight: number;
            /** Contribution */
            contribution: number;
            /** Reason */
            reason: string;
        };
        /** Phase6BFusedCandidateSummary */
        Phase6BFusedCandidateSummary: {
            /** Cluster Id */
            cluster_id: string;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Radius Km */
            radius_km: number;
            /** Relative Rank Score */
            relative_rank_score: number;
            /**
             * Score Semantics
             * @default uncalibrated_relative_rank_not_probability
             * @constant
             */
            score_semantics: "uncalibrated_relative_rank_not_probability";
            /** Provider Count */
            provider_count: number;
            /** Independent Family Count */
            independent_family_count: number;
            /** Same Family Duplicate Support */
            same_family_duplicate_support: number;
            /** Spread Km */
            spread_km: number;
            /** Members */
            members: components["schemas"]["Phase6BFusionMemberSummary"][];
            /** Contributions */
            contributions: components["schemas"]["Phase6BFusionContributionSummary"][];
            /** Ocr Agreement */
            ocr_agreement: boolean;
            /** Ocr Contradiction */
            ocr_contradiction: boolean;
        };
        /** Phase6BAgreementSummary */
        Phase6BAgreementSummary: {
            /** Provider Count */
            provider_count: number;
            /** Independent Family Count */
            independent_family_count: number;
            /** Same Family Duplicate Support */
            same_family_duplicate_support: number;
            /** Geographic Disagreement */
            geographic_disagreement: boolean;
            /** Ocr Agreement */
            ocr_agreement: boolean;
            /** Ocr Contradiction */
            ocr_contradiction: boolean;
        };
        /** Phase6BFusionSummary */
        Phase6BFusionSummary: {
            /**
             * Version
             * @default phase6b-v1
             * @constant
             */
            version: "phase6b-v1";
            /** Source Families */
            source_families?: string[];
            agreement_summary: components["schemas"]["Phase6BAgreementSummary"];
            /** Candidate Clusters */
            candidate_clusters?: components["schemas"]["Phase6BFusedCandidateSummary"][];
            /** Provider Failures */
            provider_failures?: string[];
            /** Warnings */
            warnings?: string[];
        };
        /** Phase6BOCRDetectionSummary */
        Phase6BOCRDetectionSummary: {
            /** Redacted Text */
            redacted_text: string;
            /** Script */
            script: string;
            /** Confidence */
            confidence: number;
            /** Provider */
            provider: string;
            /** Profile */
            profile: string;
        };
        /** Phase6BOCRSummary */
        Phase6BOCRSummary: {
            /** Provider */
            provider: string;
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "abstained" | "skipped" | "failed";
            /** Detections */
            detections?: components["schemas"]["Phase6BOCRDetectionSummary"][];
            /** Place Evidence */
            place_evidence?: components["schemas"]["PlaceEvidenceSummary"][];
            /**
             * Fallback Used
             * @default false
             */
            fallback_used: boolean;
            /** Reason Code */
            reason_code?: string | null;
        };
        /** Phase6BCloudCandidateAdjustment */
        Phase6BCloudCandidateAdjustment: {
            /** Candidate Id */
            candidate_id: string;
            /** Adjustment */
            adjustment: number;
            /** Reason */
            reason: string;
        };
        /** Phase6BCloudObservedClue */
        Phase6BCloudObservedClue: {
            /**
             * Type
             * @enum {string}
             */
            type: "text" | "road" | "sign" | "architecture" | "terrain" | "vehicle" | "other";
            /** Observation */
            observation: string;
            /** Supports Candidate Ids */
            supports_candidate_ids?: string[];
            /** Contradicts Candidate Ids */
            contradicts_candidate_ids?: string[];
        };
        /** Phase6BCloudReviewSummary */
        Phase6BCloudReviewSummary: {
            /**
             * Decision
             * @enum {string}
             */
            decision: "support_candidate" | "reject_all" | "insufficient";
            /** Selected Candidate Ids */
            selected_candidate_ids?: string[];
            /** Candidate Adjustments */
            candidate_adjustments?: components["schemas"]["Phase6BCloudCandidateAdjustment"][];
            /** Observed Clues */
            observed_clues?: components["schemas"]["Phase6BCloudObservedClue"][];
            /** Suggested Place Query */
            suggested_place_query?: string | null;
            /** Uncertainty Reason */
            uncertainty_reason: string;
            /**
             * Requires High Detail
             * @default false
             */
            requires_high_detail: boolean;
        };
        /** Phase6BCloudBudgetSummary */
        Phase6BCloudBudgetSummary: {
            /** Calls Today */
            calls_today: number;
            /** Estimated Month Spend Usd */
            estimated_month_spend_usd: number;
            /** Configured Monthly Budget Usd */
            configured_monthly_budget_usd: number;
            /** Remaining Budget Usd */
            remaining_budget_usd: number;
        };
        /** Phase6BCloudAssistSummary */
        Phase6BCloudAssistSummary: {
            /** Allowed */
            allowed: boolean;
            /** Triggered */
            triggered: boolean;
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "skipped" | "refused" | "failed";
            /**
             * Provider
             * @default openai
             * @constant
             */
            provider: "openai";
            /** Model */
            model: string;
            /**
             * Prompt Version
             * @default openai-geo-review-v1
             * @constant
             */
            prompt_version: "openai-geo-review-v1";
            /** Reason */
            reason?: string | null;
            /** Trigger Reasons */
            trigger_reasons?: string[];
            /**
             * Cache Hit
             * @default false
             */
            cache_hit: boolean;
            /** Estimated Cost Usd */
            estimated_cost_usd?: number | null;
            /** Cost Estimate Version */
            cost_estimate_version?: string | null;
            budget?: components["schemas"]["Phase6BCloudBudgetSummary"] | null;
            review?: components["schemas"]["Phase6BCloudReviewSummary"] | null;
            /** Warnings */
            warnings?: string[];
        };
        /** GeoClipClusterSummary */
        GeoClipClusterSummary: {
            /** Cluster Id */
            cluster_id: string;
            /**
             * Source
             * @default geoclip
             * @constant
             */
            source: "geoclip";
            /** Member Count */
            member_count: number;
            /** Member Ranks */
            member_ranks: number[];
            /** Max Raw Similarity */
            max_raw_similarity: number;
            /** Mean Raw Similarity */
            mean_raw_similarity: number;
            /** Raw Score Type */
            raw_score_type: string;
            /** Cluster Support */
            cluster_support: number;
            /**
             * Score Semantics
             * @default uncalibrated_relative_rank
             * @constant
             */
            score_semantics: "uncalibrated_relative_rank";
        };
        /** UncalibratedConfidenceSummary */
        UncalibratedConfidenceSummary: {
            /**
             * Label
             * @enum {string}
             */
            label: "low" | "medium" | "high" | "very_high";
            /** Score */
            score?: null;
            /**
             * Calibrated
             * @default false
             * @constant
             */
            calibrated: false;
            /** Basis */
            basis: string[];
        };
        /** ReverseGeocodeSummary */
        ReverseGeocodeSummary: {
            /** Country */
            country?: string | null;
            /** Country Code */
            country_code?: string | null;
            /** Region */
            region?: string | null;
            /** City */
            city?: string | null;
            /** District */
            district?: string | null;
            /** Display Name */
            display_name: string;
            /** Provider */
            provider: string;
            /** Dataset Version */
            dataset_version: string;
            /** License */
            license: string;
        };
        /** PlaceLabelDiagnostics */
        PlaceLabelDiagnostics: {
            /** Country */
            country?: string | null;
            /** Region */
            region?: string | null;
            /** City */
            city?: string | null;
            /** Distance To Place Km */
            distance_to_place_km: number;
            /** Source */
            source: string;
            /** Dataset Version */
            dataset_version: string;
            /** License */
            license: string;
        };
        /** ModelPredictionDiagnostics */
        ModelPredictionDiagnostics: {
            /** Provider Id */
            provider_id: string;
            /** Model Name */
            model_name: string;
            /** Model Revision */
            model_revision: string;
            /** Implementation Revision */
            implementation_revision: string;
            /** Device */
            device: string;
            /** Dtype */
            dtype: string;
            /** Raw Score */
            raw_score: number;
            /** Score Type */
            score_type: string;
            /** Normalization Method */
            normalization_method: string;
            /**
             * Calibration State
             * @enum {string}
             */
            calibration_state: "uncalibrated" | "preliminary" | "calibrated";
            /** Original Rank */
            original_rank: number;
            /** Inference Ms */
            inference_ms: number;
            /**
             * External Transfer
             * @default false
             * @constant
             */
            external_transfer: false;
            /** Limitations */
            limitations: string[];
            place_label?: components["schemas"]["PlaceLabelDiagnostics"] | null;
        };
        /** Phase4ScoreContribution */
        Phase4ScoreContribution: {
            /** Feature */
            feature: string;
            /** Raw Value */
            raw_value: number;
            /** Weight */
            weight: number;
            /** Contribution */
            contribution: number;
            /** Reason Code */
            reason_code: string;
        };
        /** MapConstraintSummary */
        MapConstraintSummary: {
            /** Clue */
            clue: string;
            /** Map Feature */
            map_feature: string;
            /**
             * Status
             * @enum {string}
             */
            status: "supported" | "contradicted" | "neutral" | "unknown";
            /** Reliability */
            reliability: number;
            /** Query Radius Km */
            query_radius_km: number;
            /** Provider */
            provider: string;
            /** Limitation */
            limitation?: string | null;
        };
        /** GeometryVerificationSummary */
        GeometryVerificationSummary: {
            /** Reference Id */
            reference_id: string;
            /** Provider */
            provider: string;
            /**
             * Status
             * @enum {string}
             */
            status: "supported" | "inconclusive" | "contradicted" | "unavailable" | "failed";
            /** Query Keypoints */
            query_keypoints: number;
            /** Reference Keypoints */
            reference_keypoints: number;
            /** Raw Matches */
            raw_matches: number;
            /** Filtered Matches */
            filtered_matches: number;
            /** Inliers */
            inliers: number;
            /** Inlier Ratio */
            inlier_ratio: number;
            /** Query Coverage */
            query_coverage: number;
            /** Reference Coverage */
            reference_coverage: number;
            /** Residual Error Px */
            residual_error_px?: number | null;
            /** Robust Model Type */
            robust_model_type?: ("homography" | "fundamental_matrix") | null;
            /** Limitations */
            limitations?: string[];
            /** Runtime Ms */
            runtime_ms: number;
        };
        /** ReferenceAttribution */
        ReferenceAttribution: {
            /** Reference Id */
            reference_id: string;
            /** Source */
            source: string;
            /** License */
            license: string;
            /** Attribution */
            attribution: string;
            /**
             * Display Allowed
             * @default false
             */
            display_allowed: boolean;
        };
        /** Phase4Assessment */
        Phase4Assessment: {
            /**
             * Classification
             * @enum {string}
             */
            classification: "model_only" | "retrieval_only" | "map_supported" | "geometry_supported" | "multi_source_supported" | "contradicted" | "abstained";
            /** Relative Rank Score */
            relative_rank_score: number;
            /**
             * Score Semantics
             * @default uncalibrated_relative_rank
             * @constant
             */
            score_semantics: "uncalibrated_relative_rank";
            /** Reranker Version */
            reranker_version: string;
            /** Score Breakdown */
            score_breakdown: components["schemas"]["Phase4ScoreContribution"][];
            /** Source Diversity */
            source_diversity: number;
            /** Contributing Retrieval Hit Ids */
            contributing_retrieval_hit_ids: string[];
            /** Map Observations */
            map_observations?: components["schemas"]["MapConstraintSummary"][];
            /** Geometry Results */
            geometry_results?: components["schemas"]["GeometryVerificationSummary"][];
            /** Contradictions */
            contradictions?: string[];
            /** Reference Attributions */
            reference_attributions?: components["schemas"]["ReferenceAttribution"][];
            /** Limitations */
            limitations?: string[];
        };
        /** Phase5BScoreContribution */
        Phase5BScoreContribution: {
            /** Feature */
            feature: string;
            /** Raw Value */
            raw_value: number;
            /** Weight */
            weight: number;
            /** Contribution */
            contribution: number;
            /** Reason Code */
            reason_code: string;
        };
        /** PlaceEvidenceSummary */
        PlaceEvidenceSummary: {
            /** Matched Entity */
            matched_entity: string;
            /** Normalized Name */
            normalized_name: string;
            /** Country Code */
            country_code?: string | null;
            /** Region */
            region?: string | null;
            center: components["schemas"]["GeoPoint"];
            /**
             * Match Type
             * @enum {string}
             */
            match_type: "country" | "region" | "city" | "district" | "road" | "airport" | "station" | "public_landmark" | "public_institution" | "domain_suffix";
            /** Text Similarity */
            text_similarity: number;
            /** Ambiguity Count */
            ambiguity_count: number;
            /** Evidence Strength */
            evidence_strength: number;
            /** Source */
            source: string;
            /** Dataset Version */
            dataset_version: string;
            /** License */
            license: string;
        };
        /** RetrievalMatchSummary */
        RetrievalMatchSummary: {
            /** Reference Id */
            reference_id: string;
            /** Provider */
            provider: string;
            /** Source */
            source: string;
            /** Distance */
            distance: number;
            /** Relative Similarity */
            relative_similarity: number;
            center: components["schemas"]["GeoPoint"];
            /** Geographic Cluster */
            geographic_cluster: string;
            /** License */
            license: string;
            /** Attribution */
            attribution: string;
            /**
             * Display Allowed
             * @default false
             */
            display_allowed: boolean;
        };
        /** Phase5BAssessment */
        Phase5BAssessment: {
            /**
             * Classification
             * @enum {string}
             */
            classification: "model_only" | "place_supported" | "retrieval_supported" | "map_supported" | "multi_source_supported" | "contradicted";
            /** Relative Rank Score */
            relative_rank_score: number;
            /**
             * Score Semantics
             * @default uncalibrated_relative_rank
             * @constant
             */
            score_semantics: "uncalibrated_relative_rank";
            /**
             * Reranker Version
             * @default phase5b-v1
             * @enum {string}
             */
            reranker_version: "phase5b-v1" | "phase6a-v1" | "phase6b-v1" | "phase6c-v1";
            /** Score Breakdown */
            score_breakdown: components["schemas"]["Phase5BScoreContribution"][];
            /** Provider Diversity */
            provider_diversity: number;
            /** Source Diversity */
            source_diversity: number;
            /** Place Matches */
            place_matches?: components["schemas"]["PlaceEvidenceSummary"][];
            /** Retrieval Matches */
            retrieval_matches?: components["schemas"]["RetrievalMatchSummary"][];
            /** Map Observations */
            map_observations?: components["schemas"]["MapConstraintSummary"][];
            /** Supports */
            supports?: string[];
            /** Contradictions */
            contradictions?: string[];
            /** Movement Reasons */
            movement_reasons?: string[];
            /** Limitations */
            limitations?: string[];
        };
        /** ProviderRunDiagnostic */
        ProviderRunDiagnostic: {
            /** Provider Id */
            provider_id: string;
            /** Provider Type */
            provider_type: string;
            /**
             * Status
             * @enum {string}
             */
            status: "succeeded" | "abstained" | "skipped" | "failed";
            /** Duration Ms */
            duration_ms: number;
            /** Reason Code */
            reason_code?: string | null;
            /** Device */
            device?: string | null;
            /** Offline */
            offline: boolean;
        };
        /** ReferenceIndexDiagnostic */
        ReferenceIndexDiagnostic: {
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "unavailable" | "disabled" | "incompatible";
            /** Index Id */
            index_id?: string | null;
            /** Embedding Provider */
            embedding_provider?: string | null;
            /** Embedding Version */
            embedding_version?: string | null;
            /** Dimension */
            dimension?: number | null;
            /** Image Count */
            image_count: number;
            /** Checksum */
            checksum?: string | null;
        };
        /** Phase5BDiagnostics */
        Phase5BDiagnostics: {
            /**
             * Reranker Version
             * @default phase5b-v1
             * @enum {string}
             */
            reranker_version: "phase5b-v1" | "phase6a-v1" | "phase6b-v1";
            /** Providers */
            providers: components["schemas"]["ProviderRunDiagnostic"][];
            reference_index?: components["schemas"]["ReferenceIndexDiagnostic"] | null;
            /** Partial Failures */
            partial_failures?: string[];
        };
        /** Abstention */
        Abstention: {
            /**
             * Abstained
             * @default true
             * @constant
             */
            abstained: true;
            /** Reason Code */
            reason_code: string;
            /** Message Key */
            message_key: string;
        };
        /** FailureSummary */
        FailureSummary: {
            /** Code */
            code: string;
            /** Message Key */
            message_key: string;
            /** Retryable */
            retryable: boolean;
        };
        /** SimulationSummary */
        SimulationSummary: {
            /** Scenario Id */
            scenario_id: string;
            /**
             * Warning Key
             * @default warning.simulated_development_result
             * @constant
             */
            warning_key: "warning.simulated_development_result";
            /**
             * Watermark
             * @default SIMULATED DEVELOPMENT RESULT
             * @constant
             */
            watermark: "SIMULATED DEVELOPMENT RESULT";
        };
        /** ProviderComparisonSummary */
        ProviderComparisonSummary: {
            /** Provider Id */
            provider_id: string;
            /**
             * Mode
             * @enum {string}
             */
            mode: "shadow" | "candidate";
            /**
             * Status
             * @enum {string}
             */
            status: "succeeded" | "abstained" | "skipped" | "failed";
            /** Runtime Ms */
            runtime_ms: number;
            /** Distance To Primary Km */
            distance_to_primary_km?: number | null;
            /** Candidate Overlap */
            candidate_overlap?: number | null;
            /**
             * Ranking Impact
             * @default none
             * @enum {string}
             */
            ranking_impact: "none" | "eligible_not_applied";
            /** Failure Code */
            failure_code?: string | null;
        };
        /** Analysis */
        Analysis: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            status: components["schemas"]["AnalysisStatus"];
            analysis_mode: components["schemas"]["AnalysisMode"];
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Expires At */
            expires_at: string | null;
            progress: components["schemas"]["Progress"];
            image?: components["schemas"]["ImageSummary"] | null;
            quality?: components["schemas"]["QualitySummary"] | null;
            /** Evidence */
            evidence: components["schemas"]["Evidence"][];
            /** Candidates */
            candidates: components["schemas"]["Candidate"][];
            abstention?: components["schemas"]["Abstention"] | null;
            /** Warnings */
            warnings: string[];
            /** Timings Ms */
            timings_ms: {
                [key: string]: number;
            };
            /** Fusion Policy Version */
            fusion_policy_version: string;
            /**
             * Pipeline Version
             * @default legacy-v1
             */
            pipeline_version: string;
            phase5b_diagnostics?: components["schemas"]["Phase5BDiagnostics"] | null;
            scene_analysis?: components["schemas"]["SceneSegmentationSummary"] | null;
            /** Model Predictions */
            model_predictions?: {
                [key: string]: components["schemas"]["Phase6BProviderPredictionSummary"];
            } | null;
            fusion?: components["schemas"]["Phase6BFusionSummary"] | null;
            ocr?: components["schemas"]["Phase6BOCRSummary"] | null;
            cloud_assist?: components["schemas"]["Phase6BCloudAssistSummary"] | null;
            phase6c?: components["schemas"]["Phase6CAnalysisSummary"] | null;
            /**
             * Result Classification
             * @default real
             * @enum {string}
             */
            result_classification: "real" | "simulated";
            simulation?: components["schemas"]["SimulationSummary"] | null;
            /** Provider Comparisons */
            provider_comparisons?: components["schemas"]["ProviderComparisonSummary"][];
            failure?: components["schemas"]["FailureSummary"] | null;
        };
        /** AnalysisHistoryItem */
        AnalysisHistoryItem: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Expires At */
            expires_at: string | null;
            status: components["schemas"]["AnalysisStatus"];
            analysis_mode: components["schemas"]["AnalysisMode"];
            /**
             * Result Classification
             * @enum {string}
             */
            result_classification: "real" | "simulated";
            /** Image Sha256 */
            image_sha256?: string | null;
            /** Image Width */
            image_width?: number | null;
            /** Image Height */
            image_height?: number | null;
            /** Provider Ids */
            provider_ids: string[];
            /** Primary Label */
            primary_label?: string | null;
            /** Candidate Count */
            candidate_count: number;
            /** Evidence Count */
            evidence_count: number;
            /** Runtime Ms */
            runtime_ms?: number | null;
            /** Warning Count */
            warning_count: number;
            /** Source Retained */
            source_retained: boolean;
            /**
             * Deletion State
             * @default active
             * @constant
             */
            deletion_state: "active";
        };
        /** AnalysisHistoryPage */
        AnalysisHistoryPage: {
            /** Items */
            items: components["schemas"]["AnalysisHistoryItem"][];
            /** Total */
            total: number;
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
        };
        /** RatioView */
        RatioView: {
            /** Numerator */
            numerator: number;
            /** Denominator */
            denominator: number;
            /** Value */
            value?: number | null;
        };
        /** EvaluationReportSummary */
        EvaluationReportSummary: {
            /** Report Id */
            report_id: string;
            /** Provider Id */
            provider_id: string;
            /** Model Revision */
            model_revision: string;
            /** Evaluation Fingerprint */
            evaluation_fingerprint: string;
            /** Image Count */
            image_count: number;
            /**
             * Calibration State
             * @enum {string}
             */
            calibration_state: "uncalibrated" | "preliminary" | "calibrated";
            country_top1: components["schemas"]["RatioView"];
            country_top5: components["schemas"]["RatioView"];
            region_top1: components["schemas"]["RatioView"];
            city_top1: components["schemas"]["RatioView"];
            /** Recall Top1 */
            recall_top1: {
                [key: string]: components["schemas"]["RatioView"];
            };
            /** Mean Error Km */
            mean_error_km?: number | null;
            /** Median Error Km */
            median_error_km?: number | null;
            /** P95 Error Km */
            p95_error_km?: number | null;
            abstention: components["schemas"]["RatioView"];
            provider_failure: components["schemas"]["RatioView"];
            /** Latency Median Ms */
            latency_median_ms?: number | null;
            /** Latency P95 Ms */
            latency_p95_ms?: number | null;
            uncertainty_coverage: components["schemas"]["RatioView"];
            /** Geographic Distribution */
            geographic_distribution: {
                [key: string]: number;
            };
            /** Scene Distribution */
            scene_distribution: {
                [key: string]: number;
            };
            /** Exclusions */
            exclusions: {
                [key: string]: number;
            };
            /** Limitations */
            limitations: string[];
        };
        /** EvaluationReportList */
        EvaluationReportList: {
            /** Reports */
            reports: components["schemas"]["EvaluationReportSummary"][];
        };
        /** DatasetQAIssueView */
        DatasetQAIssueView: {
            /** Code */
            code: string;
            /**
             * Severity
             * @enum {string}
             */
            severity: "error" | "warning" | "info";
            /** Asset Key */
            asset_key: string;
            /** Field */
            field?: string | null;
            /** Message Key */
            message_key: string;
            /** Safe Metrics */
            safe_metrics?: {
                [key: string]: string;
            };
        };
        /** DatasetQAReportSummary */
        DatasetQAReportSummary: {
            /** Report Id */
            report_id: string;
            /**
             * Schema Version
             * @default 1
             * @constant
             */
            schema_version: 1;
            /** Dataset Fingerprint */
            dataset_fingerprint: string;
            /**
             * Dataset Type
             * @enum {string}
             */
            dataset_type: "geolocation" | "segmentation" | "mixed";
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Scanned Images */
            scanned_images: number;
            /** Scanned Masks */
            scanned_masks: number;
            /** Error Count */
            error_count: number;
            /** Warning Count */
            warning_count: number;
        };
        /** DatasetQAReport */
        DatasetQAReport: {
            summary: components["schemas"]["DatasetQAReportSummary"];
            /** Checks */
            checks: {
                [key: string]: "passed" | "failed" | "warning" | "unavailable";
            };
            /** Distributions */
            distributions: {
                [key: string]: {
                    [key: string]: number;
                };
            };
            /** Issues */
            issues: components["schemas"]["DatasetQAIssueView"][];
            /** Outputs */
            outputs: string[];
            /** Limitations */
            limitations: string[];
        };
        /** DatasetQAReportList */
        DatasetQAReportList: {
            /** Reports */
            reports: components["schemas"]["DatasetQAReportSummary"][];
        };
        AnalysisEvent: {
            event_id: string;
            /** @enum {string} */
            event_type: "progress" | "heartbeat" | "completed" | "failed" | "deleted";
            /** Format: uuid */
            analysis_id: string;
            /** Format: date-time */
            occurred_at: string;
            status: components["schemas"]["AnalysisStatus"];
            progress?: components["schemas"]["Progress"] | null;
        };
        /** DeleteResponse */
        DeleteResponse: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Deleted
             * @default true
             * @constant
             */
            deleted: true;
        };
        ProblemDetails: {
            type: string;
            title: string;
            status: number;
            code: string;
            message_key: string;
            request_id: string;
            retry_after_seconds?: number | null;
        };
        /** Body_create_analysis_api_v1_analyses_post */
        Body_create_analysis_api_v1_analyses_post: {
            /**
             * Image
             * Format: binary
             */
            image: string;
            analysis_mode: components["schemas"]["AnalysisMode"];
            /** Cloud Processing Consent */
            cloud_processing_consent: boolean;
            /** Authorization Acknowledged */
            authorization_acknowledged: boolean;
            /**
             * Allow Cloud Assist
             * @default false
             */
            allow_cloud_assist: boolean;
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /** Phase6CAblationCandidateSummary */
        Phase6CAblationCandidateSummary: {
            /** Rank */
            rank: number;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Uncertainty Radius Km */
            uncertainty_radius_km: number;
            /** Relative Rank Score */
            relative_rank_score: number;
            /**
             * Score Semantics
             * @default uncalibrated_relative_rank_not_probability
             * @constant
             */
            score_semantics: "uncalibrated_relative_rank_not_probability";
            /** Publication Eligible */
            publication_eligible: boolean;
            /** Evidence Kinds */
            evidence_kinds: string[];
        };
        /** Phase6CAblationSummary */
        Phase6CAblationSummary: {
            /** Profile Id */
            profile_id: string;
            /** Candidate Count */
            candidate_count: number;
            /** Publication Candidate Count */
            publication_candidate_count: number;
            /** Abstained */
            abstained: boolean;
            /** Abstention Reason */
            abstention_reason?: string | null;
            /** Included Evidence Kinds */
            included_evidence_kinds: string[];
            /** Excluded Evidence Kinds */
            excluded_evidence_kinds: string[];
            /** Candidates */
            candidates?: components["schemas"]["Phase6CAblationCandidateSummary"][];
        };
        /** Phase6CAnalysisSummary */
        Phase6CAnalysisSummary: {
            /**
             * Schema Version
             * @default atlaslens-phase6c-analysis-v1
             * @constant
             */
            schema_version: "atlaslens-phase6c-analysis-v1";
            /**
             * Pipeline Version
             * @default phase6c-v1
             * @constant
             */
            pipeline_version: "phase6c-v1";
            /**
             * Fusion Version
             * @default phase6c-v1
             * @constant
             */
            fusion_version: "phase6c-v1";
            /** Reference Index Version */
            reference_index_version?: string | null;
            /** Evaluation Run Id */
            evaluation_run_id?: string | null;
            /** Hierarchical Candidates */
            hierarchical_candidates?: components["schemas"]["Phase6CHierarchicalCandidateSummary"][];
            /** Megaloc Matches */
            megaloc_matches?: components["schemas"]["Phase6CMegaLocMatchSummary"][];
            /** G3 Scores */
            g3_scores?: components["schemas"]["Phase6CG3ScoreSummary"][];
            /** Providers */
            providers?: components["schemas"]["Phase6CProviderRunSummary"][];
            /** Fusion Candidates */
            fusion_candidates?: components["schemas"]["Phase6CFusedCandidateSummary"][];
            /**
             * Publication Candidate Count
             * @default 0
             */
            publication_candidate_count: number;
            /**
             * Turkiye Signal Count
             * @default 0
             */
            turkiye_signal_count: number;
            /** Turkiye Signal Groups */
            turkiye_signal_groups?: string[];
            leakage_audit: components["schemas"]["Phase6CLeakageAuditSummary"];
            /** Reference Attributions */
            reference_attributions?: string[];
            /** Ablations */
            ablations?: components["schemas"]["Phase6CAblationSummary"][];
            /** Cache Fingerprint */
            cache_fingerprint: string;
        };
        /** Phase6CFusedCandidateSummary */
        Phase6CFusedCandidateSummary: {
            /** Cluster Id */
            cluster_id: string;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Uncertainty Radius Km */
            uncertainty_radius_km: number;
            /** Relative Rank Score */
            relative_rank_score: number;
            /**
             * Score Semantics
             * @default uncalibrated_relative_rank_not_probability
             * @constant
             */
            score_semantics: "uncalibrated_relative_rank_not_probability";
            /** Confidence */
            confidence?: null;
            /**
             * Calibrated
             * @default false
             * @constant
             */
            calibrated: false;
            /** Source Family Count */
            source_family_count: number;
            /** Independent Source Family Count */
            independent_source_family_count: number;
            /** Correlated Source Family Count */
            correlated_source_family_count: number;
            /** Provider Count */
            provider_count: number;
            /** Spread Km */
            spread_km: number;
            /** Publication Eligible */
            publication_eligible: boolean;
            /**
             * Publication Basis
             * @enum {string}
             */
            publication_basis: "at_least_two_independent_source_families" | "candidate_recall_only_insufficient_independence";
            /** Pre Diversity Rank */
            pre_diversity_rank: number;
            /** Final Rank */
            final_rank: number;
            /** Rank Movement */
            rank_movement: number;
            /** Movement Reasons */
            movement_reasons: string[];
            /** Members */
            members: components["schemas"]["Phase6CFusionMemberSummary"][];
            /** Contributions */
            contributions: components["schemas"]["Phase6CFusionContributionSummary"][];
        };
        /** Phase6CFusionContributionSummary */
        Phase6CFusionContributionSummary: {
            /** Name */
            name: string;
            /** Source */
            source: string;
            /** Raw Value */
            raw_value: number;
            /** Normalized Value */
            normalized_value: number;
            /** Weight */
            weight: number;
            /** Contribution */
            contribution: number;
            /** Reason */
            reason: string;
            /** Independent */
            independent: boolean;
            /** Correlation Group */
            correlation_group: string;
        };
        /** Phase6CFusionMemberSummary */
        Phase6CFusionMemberSummary: {
            /**
             * Evidence Kind
             * @enum {string}
             */
            evidence_kind: "geoclip_original" | "geoclip_hierarchical" | "osv_direct_regression" | "plonk_samples" | "megaloc_retrieval" | "g3_verification" | "ocr_place_match" | "openai_review";
            /** Provider */
            provider: string;
            /** Model Id */
            model_id: string;
            /** Model Revision */
            model_revision: string;
            /** Candidate Id */
            candidate_id: string;
            /** Source Family */
            source_family: string;
            /** Correlation Group */
            correlation_group: string;
            /** Score Semantics */
            score_semantics: string;
            /** Raw Value */
            raw_value?: number | null;
            /** Provider Rank */
            provider_rank: number;
            /** Sample Support */
            sample_support: number;
            /** Provenance */
            provenance: string;
        };
        /** Phase6CG3ScoreSummary */
        Phase6CG3ScoreSummary: {
            /** Candidate Id */
            candidate_id: string;
            /** Rank */
            rank: number;
            /** Raw Score */
            raw_score: number;
            /**
             * Score Semantics
             * @default raw_g3_similarity_not_confidence
             * @constant
             */
            score_semantics: "raw_g3_similarity_not_confidence";
        };
        /** Phase6CHierarchicalCandidateSummary */
        Phase6CHierarchicalCandidateSummary: {
            /** Candidate Id */
            candidate_id: string;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /**
             * Search Level
             * @enum {string}
             */
            search_level: "global" | "catalogue" | "regional_refinement" | "turkiye_refinement";
            /** Provider Rank */
            provider_rank: number;
            /** Provider Score */
            provider_score: number;
            /**
             * Score Semantics
             * @default raw_cosine_similarity_not_confidence
             * @constant
             */
            score_semantics: "raw_cosine_similarity_not_confidence";
            /** Grid Resolution Km */
            grid_resolution_km: number;
            /** Diversity Cluster */
            diversity_cluster: string;
            /** Nearest Name */
            nearest_name?: string | null;
            /** Nearest Kind */
            nearest_kind?: string | null;
            /** Nearest Distance Km */
            nearest_distance_km?: number | null;
        };
        /** Phase6CLeakageAuditSummary */
        Phase6CLeakageAuditSummary: {
            /**
             * Status
             * @enum {string}
             */
            status: "passed" | "failed" | "not_run" | "incomplete";
            /** Audit Version */
            audit_version: string;
            /** Report Fingerprint */
            report_fingerprint?: string | null;
            /** References Checked */
            references_checked: number;
            /** References Excluded */
            references_excluded: number;
            /** Reason Code */
            reason_code?: string | null;
        };
        /** Phase6CMegaLocMatchSummary */
        Phase6CMegaLocMatchSummary: {
            /** Reference Id */
            reference_id: string;
            /** Rank */
            rank: number;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Similarity */
            similarity: number;
            /**
             * Similarity Semantics
             * @default cosine_similarity_not_confidence
             * @constant
             */
            similarity_semantics: "cosine_similarity_not_confidence";
            /** Confidence */
            confidence?: null;
            /** Uncertainty Radius M */
            uncertainty_radius_m: number;
            /**
             * Source
             * @enum {string}
             */
            source: "mapillary" | "kartaview" | "manual";
            /** Source Family */
            source_family: string;
            /** Source Sequence Id */
            source_sequence_id: string;
            /** Source Url */
            source_url: string;
            /** Captured At */
            captured_at?: string | null;
            /** Province */
            province: string;
            /** City */
            city?: string | null;
            /** License */
            license: string;
            /** Attribution */
            attribution: string;
        };
        /** Phase6CProviderRunSummary */
        Phase6CProviderRunSummary: {
            /** Provider Id */
            provider_id: string;
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "abstained" | "skipped" | "disabled" | "unavailable" | "failed" | "timeout";
            /** Source Revision */
            source_revision?: string | null;
            /** Model Revision */
            model_revision?: string | null;
            /** Duration Ms */
            duration_ms: number;
            /** Candidates Produced */
            candidates_produced: number;
            /** Reason Code */
            reason_code?: string | null;
        };
        /** SystemIntelligenceLeakageAudit */
        SystemIntelligenceLeakageAudit: {
            /**
             * Status
             * @default passed
             * @constant
             */
            status: "passed";
            /**
             * Audit Version
             * @default atlaslens-leakage-audit-v1
             * @constant
             */
            audit_version: "atlaslens-leakage-audit-v1";
            /** Audit Fingerprint */
            audit_fingerprint: string;
            /** Source Report Sha256 */
            source_report_sha256: string;
            /** Checked Reference Count */
            checked_reference_count: number;
            /** Descriptor Checked Count */
            descriptor_checked_count: number;
            /** Excluded Reference Count */
            excluded_reference_count: number;
            /** Pre Index Excluded Reference Count */
            pre_index_excluded_reference_count?: number | null;
        };
        /** SystemIntelligenceModelCard */
        SystemIntelligenceModelCard: {
            /** Model Id */
            model_id: string;
            /** Display Name */
            display_name: string;
            /** Runtime Model Id */
            runtime_model_id?: string | null;
            /** Repository Url */
            repository_url?: string | null;
            /** Purpose */
            purpose: string;
            /** Enabled */
            enabled: boolean;
            /** Available */
            available: boolean;
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "not_installed" | "dependencies_installed" | "weights_prepared" | "worker_unreachable" | "model_load_failed" | "inference_not_verified" | "incomplete" | "loading" | "failed" | "disabled" | "unavailable";
            /** Installed */
            installed?: boolean | null;
            /** Weights Available */
            weights_available?: boolean | null;
            /** Worker Reachable */
            worker_reachable?: boolean | null;
            /** Model Loaded */
            model_loaded?: boolean | null;
            /** Load Verified */
            load_verified?: boolean | null;
            /** Real Inference Verified */
            real_inference_verified?: boolean | null;
            /** Device */
            device?: string | null;
            /**
             * Execution Mode
             * @enum {string}
             */
            execution_mode: "in_process" | "isolated_worker" | "isolated_process" | "cloud" | "not_integrated";
            /** Source Revision */
            source_revision?: string | null;
            /** Model Revision */
            model_revision?: string | null;
            /** License */
            license?: string | null;
            /** Last Success At */
            last_success_at?: string | null;
            /** Last Latency Ms */
            last_latency_ms?: number | null;
            /** Error Code */
            error_code?: string | null;
            /**
             * Current Participation
             * @enum {string}
             */
            current_participation: "primary" | "candidate" | "fallback" | "descriptive" | "verifier" | "optional_review" | "shadow" | "disabled" | "not_integrated";
        };
        /** SystemIntelligenceReferenceIndexCard */
        SystemIntelligenceReferenceIndexCard: {
            /**
             * Index Id
             * @default turkiye_megaloc_reference_index
             * @constant
             */
            index_id: "turkiye_megaloc_reference_index";
            /** Enabled */
            enabled: boolean;
            /**
             * Status
             * @enum {string}
             */
            status: "ready" | "empty" | "unavailable" | "invalid" | "disabled";
            /** Reason Code */
            reason_code: string;
            /** Index Version */
            index_version?: string | null;
            /** Descriptor Version */
            descriptor_version?: string | null;
            /** Count */
            count: number;
            /** Sequences */
            sequences: number;
            /** Countries */
            countries?: number | null;
            /** Provinces */
            provinces?: number | null;
            /** Images Per Province */
            images_per_province?: {
                [key: string]: number;
            } | null;
            /** Source Distribution */
            source_distribution?: {
                [key: string]: number;
            } | null;
            /** Built At */
            built_at?: string | null;
            /** Disk Usage Bytes */
            disk_usage_bytes: number;
            /**
             * Leakage Status
             * @enum {string}
             */
            leakage_status: "passed" | "failed" | "not_run" | "unavailable";
            leakage_audit?: components["schemas"]["SystemIntelligenceLeakageAudit"] | null;
            /** Duplicates */
            duplicates?: number | null;
            /** Excluded */
            excluded?: number | null;
            /** Attributions */
            attributions?: string[];
            /**
             * Health
             * @enum {string}
             */
            health: "healthy" | "degraded" | "unavailable" | "invalid" | "disabled";
        };
        /** SystemIntelligenceResponse */
        SystemIntelligenceResponse: {
            /**
             * Active Pipeline Version
             * @enum {string}
             */
            active_pipeline_version: "legacy-v1" | "phase6c-v1";
            /** Models */
            models: components["schemas"]["SystemIntelligenceModelCard"][];
            reference_index: components["schemas"]["SystemIntelligenceReferenceIndexCard"];
        };
        /** ValidationError */
        ValidationError: {
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
        /**
         * CasePurpose
         * @enum {string}
         */
        CasePurpose: "journalism" | "humanitarian" | "disaster_response" | "insurance" | "authorized_security_research" | "other";
        /**
         * CaseStatus
         * @enum {string}
         */
        CaseStatus: "open" | "under_review" | "resolved" | "archived";
        /**
         * CaseSensitivity
         * @enum {string}
         */
        CaseSensitivity: "standard" | "sensitive" | "conflict_related";
        /**
         * AdjudicationDecision
         * @enum {string}
         */
        AdjudicationDecision: "accepted" | "rejected" | "needs_more_evidence" | "withdrawn";
        /** CaseCreateRequest */
        CaseCreateRequest: {
            /** Title */
            title: string;
            /** Description */
            description?: string | null;
            purpose: components["schemas"]["CasePurpose"];
            /** Purpose Detail */
            purpose_detail?: string | null;
            /** Source Context */
            source_context: string;
            sensitivity: components["schemas"]["CaseSensitivity"];
            /**
             * Authorization Attested
             * @constant
             */
            authorization_attested: true;
            /**
             * Retention Policy
             * @description Explicit deployment policy identifier; the vocabulary is not yet production-fixed.
             */
            retention_policy: string;
            /** Created By Actor Id */
            created_by_actor_id: string;
        };
        /** CasePatchRequest */
        CasePatchRequest: {
            /** Actor Id */
            actor_id: string;
            /** Expected Version */
            expected_version: number;
            /** Title */
            title?: string | null;
            /** Description */
            description?: string | null;
            purpose?: components["schemas"]["CasePurpose"] | null;
            /** Purpose Detail */
            purpose_detail?: string | null;
            /** Source Context */
            source_context?: string | null;
            status?: components["schemas"]["CaseStatus"] | null;
            sensitivity?: components["schemas"]["CaseSensitivity"] | null;
            /** Retention Policy */
            retention_policy?: string | null;
        };
        /** CaseView */
        CaseView: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Workspace Id
             * @description Local default workspace seam only; production tenant isolation is not claimed.
             */
            workspace_id: string;
            /** Title */
            title: string;
            /** Description */
            description?: string | null;
            purpose: components["schemas"]["CasePurpose"];
            /** Purpose Detail */
            purpose_detail?: string | null;
            /** Source Context */
            source_context: string;
            status: components["schemas"]["CaseStatus"];
            sensitivity: components["schemas"]["CaseSensitivity"];
            /** Authorization Attested */
            authorization_attested: boolean;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at: string;
            /** Closed At */
            closed_at?: string | null;
            /** Created By Actor Id */
            created_by_actor_id: string;
            /**
             * Retention Policy
             * @description Applied retention-policy identifier.
             */
            retention_policy: string;
            /** Version */
            version: number;
            /** Media Count */
            media_count: number;
            /** Evidence Count */
            evidence_count: number;
            /** Hypothesis Count */
            hypothesis_count: number;
            /** Adjudication Count */
            adjudication_count: number;
            latest_adjudication_decision?: components["schemas"]["AdjudicationDecision"] | null;
        };
        /** CasePage */
        CasePage: {
            /** Items */
            items: components["schemas"]["CaseView"][];
            /** Total */
            total: number;
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /**
             * Ordering
             * @default updated_at_desc_id_desc
             * @constant
             */
            ordering: "updated_at_desc_id_desc";
        };
        /**
         * MediaType
         * @enum {string}
         */
        MediaType: "image";
        /**
         * MediaSourceType
         * @enum {string}
         */
        MediaSourceType: "upload" | "source_url" | "external_archive" | "other";
        /**
         * MediaStorageState
         * @enum {string}
         */
        MediaStorageState: "ephemeral" | "deleted_after_analysis" | "unavailable" | "externally_managed";
        /**
         * CaseMediaCreateRequest
         * @description Metadata-only image record. Upload and analysis remain on POST /api/v1/analyses.
         */
        CaseMediaCreateRequest: {
            /** Actor Id */
            actor_id: string;
            /**
             * Media Type
             * @constant
             */
            media_type: "image";
            source_type: components["schemas"]["MediaSourceType"];
            /**
             * Original Filename Display
             * @description Client display name only; the API strips paths and control characters before persistence.
             */
            original_filename_display: string;
            /**
             * Mime Type
             * @enum {string}
             */
            mime_type: "image/jpeg" | "image/png" | "image/webp";
            /** Byte Size */
            byte_size: number;
            /** Sha256 */
            sha256: string;
            /** Captured At */
            captured_at?: string | null;
            /** Source Url */
            source_url?: string | null;
            /** Archive Url */
            archive_url?: string | null;
            /** Source Description */
            source_description?: string | null;
            /**
             * Authorization Attested
             * @constant
             */
            authorization_attested: true;
            storage_state: components["schemas"]["MediaStorageState"];
        };
        /** CaseMediaView */
        CaseMediaView: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Case Id
             * Format: uuid
             */
            case_id: string;
            /** Analysis Id */
            analysis_id?: string | null;
            media_type: components["schemas"]["MediaType"];
            source_type: components["schemas"]["MediaSourceType"];
            /** Original Filename Display */
            original_filename_display: string;
            /**
             * Mime Type
             * @enum {string}
             */
            mime_type: "image/jpeg" | "image/png" | "image/webp";
            /** Byte Size */
            byte_size: number;
            /** Sha256 */
            sha256: string;
            /** Captured At */
            captured_at?: string | null;
            /**
             * Received At
             * Format: date-time
             */
            received_at: string;
            /** Source Url */
            source_url?: string | null;
            /** Archive Url */
            archive_url?: string | null;
            /** Source Description */
            source_description?: string | null;
            /** Authorization Attested */
            authorization_attested: boolean;
            storage_state: components["schemas"]["MediaStorageState"];
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Analysis Status At Link */
            analysis_status_at_link?: string | null;
            /** Analysis Created At */
            analysis_created_at?: string | null;
            /** Analysis Expires At */
            analysis_expires_at?: string | null;
            /** Materialized At */
            materialized_at?: string | null;
        };
        /** CaseMediaPage */
        CaseMediaPage: {
            /** Items */
            items: components["schemas"]["CaseMediaView"][];
            /** Total */
            total: number;
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /**
             * Ordering
             * @default created_at_asc_id_asc
             * @constant
             */
            ordering: "created_at_asc_id_asc";
        };
        /** AnalysisLinkRequest */
        AnalysisLinkRequest: {
            /**
             * Media Id
             * Format: uuid
             */
            media_id: string;
            /** Actor Id */
            actor_id: string;
        };
        /** MaterializeEvidenceRequest */
        MaterializeEvidenceRequest: {
            /** Actor Id */
            actor_id: string;
            /**
             * Media Id
             * Format: uuid
             */
            media_id: string;
        };
        /**
         * MaterializationView
         * @description Idempotent normalization result; created counts are zero on a repeated request.
         */
        MaterializationView: {
            /**
             * Case Id
             * Format: uuid
             */
            case_id: string;
            /**
             * Media Id
             * Format: uuid
             */
            media_id: string;
            /**
             * Analysis Id
             * Format: uuid
             */
            analysis_id: string;
            /** Created */
            created: boolean;
            /** Evidence Created */
            evidence_created: number;
            /** Hypotheses Created */
            hypotheses_created: number;
            /** Already Materialized */
            already_materialized: boolean;
            /** Evidence Ids */
            evidence_ids: string[];
            /** Hypothesis Ids */
            hypothesis_ids: string[];
        };
        /** @description Bounded flat JSON scalar used by current Phase 2 evidence and audit payloads; sensitive/raw media, OCR, path, prompt, and secret keys are rejected at runtime. */
        JsonValue: string | number | boolean | null;
        /**
         * EvidenceType
         * @enum {string}
         */
        EvidenceType: "metadata" | "ocr" | "visual_clue" | "model_hypothesis" | "retrieval_match" | "map_evidence" | "analyst_note";
        /**
         * EvidenceView
         * @description Immutable normalized evidence. Retrieval matches remain unverified evidence, not confirmed locations.
         */
        EvidenceView: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Case Id
             * Format: uuid
             */
            case_id: string;
            /**
             * Media Id
             * Format: uuid
             */
            media_id: string;
            /**
             * Analysis Id
             * Format: uuid
             */
            analysis_id: string;
            evidence_type: components["schemas"]["EvidenceType"];
            /** Provider */
            provider: string;
            /** Provider Family */
            provider_family: string;
            /** Summary */
            summary: string;
            /** Structured Payload */
            structured_payload: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Provenance */
            provenance: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /**
             * Observed At
             * Format: date-time
             */
            observed_at: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Immutable Source Hash */
            immutable_source_hash: string;
        };
        /** EvidencePage */
        EvidencePage: {
            /** Items */
            items: components["schemas"]["EvidenceView"][];
            /** Total */
            total: number;
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /**
             * Ordering
             * @default created_at_asc_id_asc
             * @constant
             */
            ordering: "created_at_asc_id_asc";
        };
        /**
         * HypothesisOrigin
         * @enum {string}
         */
        HypothesisOrigin: "model" | "operator_correction" | "imported";
        /**
         * CalibrationState
         * @enum {string}
         */
        CalibrationState: "calibrated" | "uncalibrated" | "not_applicable";
        /**
         * HypothesisView
         * @description A provenance-bearing location hypothesis with positive uncertainty. Model output is never overwritten.
         */
        HypothesisView: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Case Id
             * Format: uuid
             */
            case_id: string;
            /**
             * Media Id
             * Format: uuid
             */
            media_id: string;
            /**
             * Analysis Id
             * Format: uuid
             */
            analysis_id: string;
            origin: components["schemas"]["HypothesisOrigin"];
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Uncertainty Radius M */
            uncertainty_radius_m: number;
            /** Country Code */
            country_code?: string | null;
            /** Region Name */
            region_name?: string | null;
            /** Locality Name */
            locality_name?: string | null;
            /** Rank */
            rank?: number | null;
            /**
             * Confidence Label
             * @description Optional semantic label only; never an implied probability.
             */
            confidence_label?: string | null;
            calibration_state: components["schemas"]["CalibrationState"];
            /** Supporting Evidence Ids */
            supporting_evidence_ids: string[];
            /** Model Family Groups */
            model_family_groups: string[];
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Supersedes Hypothesis Id */
            supersedes_hypothesis_id?: string | null;
            /** Adjudications */
            adjudications?: components["schemas"]["AdjudicationView"][];
            /**
             * Adjudication Count
             * @default 0
             */
            adjudication_count: number;
            latest_adjudication?: components["schemas"]["AdjudicationView"] | null;
        };
        /** HypothesisPage */
        HypothesisPage: {
            /** Items */
            items: components["schemas"]["HypothesisView"][];
            /** Total */
            total: number;
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /**
             * Ordering
             * @default created_at_asc_id_asc
             * @constant
             */
            ordering: "created_at_asc_id_asc";
        };
        /** AdjudicationCreateRequest */
        AdjudicationCreateRequest: {
            /** Actor Id */
            actor_id: string;
            decision: components["schemas"]["AdjudicationDecision"];
            /** Rationale */
            rationale: string;
            /** Supersedes Adjudication Id */
            supersedes_adjudication_id?: string | null;
        };
        /**
         * AdjudicationView
         * @description Append-only analyst decision; prior records remain immutable.
         */
        AdjudicationView: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Case Id
             * Format: uuid
             */
            case_id: string;
            /**
             * Hypothesis Id
             * Format: uuid
             */
            hypothesis_id: string;
            /** Actor Id */
            actor_id: string;
            decision: components["schemas"]["AdjudicationDecision"];
            /** Rationale */
            rationale: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Supersedes Adjudication Id */
            supersedes_adjudication_id?: string | null;
        };
        /**
         * OperatorHypothesisCreateRequest
         * @description Creates a separate operator-origin hypothesis plus an initial adjudication.
         */
        OperatorHypothesisCreateRequest: {
            /** Actor Id */
            actor_id: string;
            /**
             * Media Id
             * Format: uuid
             */
            media_id: string;
            /**
             * Analysis Id
             * Format: uuid
             */
            analysis_id: string;
            /**
             * Supersedes Hypothesis Id
             * Format: uuid
             */
            supersedes_hypothesis_id: string;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Uncertainty Radius M */
            uncertainty_radius_m: number;
            /** Country Code */
            country_code?: string | null;
            /** Region Name */
            region_name?: string | null;
            /** Locality Name */
            locality_name?: string | null;
            /** Supporting Evidence Ids */
            supporting_evidence_ids?: string[];
            decision: components["schemas"]["AdjudicationDecision"];
            /** Rationale */
            rationale: string;
        };
        /** OperatorCorrectionView */
        OperatorCorrectionView: {
            hypothesis: components["schemas"]["HypothesisView"];
            adjudication: components["schemas"]["AdjudicationView"];
        };
        /**
         * AuditActorType
         * @enum {string}
         */
        AuditActorType: "operator" | "system" | "model";
        /**
         * AuditEventView
         * @description Append-only, SHA-256 hash-chained application history event with a redacted bounded payload.
         */
        AuditEventView: {
            /**
             * Id
             * Format: uuid
             */
            id: string;
            /**
             * Case Id
             * Format: uuid
             */
            case_id: string;
            /** Sequence Number */
            sequence_number: number;
            /** Event Type */
            event_type: string;
            /** Actor Id */
            actor_id: string;
            actor_type: components["schemas"]["AuditActorType"];
            /** Payload */
            payload: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Previous Event Hash */
            previous_event_hash: string;
            /** Event Hash */
            event_hash: string;
        };
        /** AuditEventPage */
        AuditEventPage: {
            /** Items */
            items: components["schemas"]["AuditEventView"][];
            /** Total */
            total: number;
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /**
             * Ordering
             * @default sequence_number_asc
             * @constant
             */
            ordering: "sequence_number_asc";
            /**
             * Integrity Scope
             * @default tamper_evident_application_history
             * @constant
             */
            integrity_scope: "tamper_evident_application_history";
        };
        /**
         * AuditIntegrityView
         * @description Tamper-evident application-history verification; not legally certified evidence.
         */
        AuditIntegrityView: {
            /**
             * Case Id
             * Format: uuid
             */
            case_id: string;
            /** Valid */
            valid: boolean;
            /** Checked Event Count */
            checked_event_count: number;
            /** First Invalid Sequence */
            first_invalid_sequence?: number | null;
            /** Reason */
            reason?: string | null;
            /**
             * Integrity Scope
             * @default tamper_evident_application_history
             * @constant
             */
            integrity_scope: "tamper_evident_application_history";
            /**
             * Legally Certified Evidence
             * @default false
             * @constant
             */
            legally_certified_evidence: false;
        };
        /**
         * MapillaryDemoAnalysisScope
         * @enum {string}
         */
        MapillaryDemoAnalysisScope: "generic_upload" | "ankara_reference_pilot";
        /**
         * MapillaryDemoCoverageStatus
         * @enum {string}
         */
        MapillaryDemoCoverageStatus: "pilot_eligible" | "insufficient" | "provider_unavailable";
        /** Body_queryMapillaryDemo */
        Body_queryMapillaryDemo: {
            /**
             * Image
             * Format: binary
             */
            image: string;
            /** Authorization Acknowledged */
            authorization_acknowledged: boolean;
            /** @default generic_upload */
            analysis_scope: components["schemas"]["MapillaryDemoAnalysisScope"];
        };
        /** MapillaryDemoCandidateResponse */
        MapillaryDemoCandidateResponse: {
            /** Rank */
            rank: number;
            /** Cosine Similarity */
            cosine_similarity: number;
            /** Cosine Distance */
            cosine_distance: number;
            /** Latitude */
            latitude: number;
            /** Longitude */
            longitude: number;
            /** Mapillary Image Id */
            mapillary_image_id: string;
            /** Contributor */
            contributor?: string | null;
            /** Source Url */
            source_url: string;
            /**
             * License Identifier
             * @default CC-BY-SA-4.0
             * @constant
             */
            license_identifier: "CC-BY-SA-4.0";
            /**
             * License Url
             * @default https://creativecommons.org/licenses/by-sa/4.0/
             * @constant
             */
            license_url: "https://creativecommons.org/licenses/by-sa/4.0/";
            /** Capture Date */
            capture_date?: string | null;
            /** Confidence */
            confidence?: null;
            /**
             * Confidence Semantics
             * @default uncalibrated_unavailable
             * @constant
             */
            confidence_semantics: "uncalibrated_unavailable";
            /**
             * Similarity Semantics
             * @default cosine_similarity_not_confidence
             * @constant
             */
            similarity_semantics: "cosine_similarity_not_confidence";
            /** Uncertainty Radius M */
            uncertainty_radius_m: number;
            /**
             * Uncertainty Semantics
             * @default presentation_radius_not_accuracy_or_probability
             * @constant
             */
            uncertainty_semantics: "presentation_radius_not_accuracy_or_probability";
            /**
             * Experimental Status
             * @default private_technical_demo_not_production
             * @constant
             */
            experimental_status: "private_technical_demo_not_production";
        };
        /** MapillaryDemoQueryResponse */
        MapillaryDemoQueryResponse: {
            /**
             * Status
             * @enum {string}
             */
            status: "completed" | "abstained" | "failed";
            /** Reason Code */
            reason_code?: string | null;
            analysis_scope: components["schemas"]["MapillaryDemoAnalysisScope"];
            coverage_status: components["schemas"]["MapillaryDemoCoverageStatus"];
            /**
             * Coverage Label
             * @default Ankara reference pilot
             * @constant
             */
            coverage_label: "Ankara reference pilot";
            /**
             * Retrieval Provider
             * @default megaloc_mapillary_faiss
             * @constant
             */
            retrieval_provider: "megaloc_mapillary_faiss";
            /**
             * Retrieval Scope
             * @default ankara_reference_collection
             * @constant
             */
            retrieval_scope: "ankara_reference_collection";
            /**
             * Result Semantics
             * @default ankara_reference_collection_visual_similarity_not_general_geolocation
             * @constant
             */
            result_semantics: "ankara_reference_collection_visual_similarity_not_general_geolocation";
            /** Abstained */
            abstained: boolean;
            /** Abstention Reason */
            abstention_reason?: string | null;
            /**
             * Similarity Semantics
             * @default cosine_similarity_not_confidence
             * @constant
             */
            similarity_semantics: "cosine_similarity_not_confidence";
            /**
             * Supported Region
             * @default Ankara pilot collection only
             * @constant
             */
            supported_region: "Ankara pilot collection only";
            /**
             * Evidence Version
             * @default phase3b3-mapillary-ankara-pilot-v1
             * @constant
             */
            evidence_version: "phase3b3-mapillary-ankara-pilot-v1";
            /**
             * Benchmark Version
             * @default phase3b3-mapillary-benchmark-v1
             * @constant
             */
            benchmark_version: "phase3b3-mapillary-benchmark-v1";
            /** City */
            city?: string | null;
            /** Index Version */
            index_version?: string | null;
            /**
             * Candidates
             * @default []
             */
            candidates: components["schemas"]["MapillaryDemoCandidateResponse"][];
            /** Confidence */
            confidence?: null;
            /**
             * Confidence Semantics
             * @default uncalibrated_unavailable
             * @constant
             */
            confidence_semantics: "uncalibrated_unavailable";
            /**
             * Experimental Status
             * @default private_technical_demo_not_production
             * @constant
             */
            experimental_status: "private_technical_demo_not_production";
            /**
             * Limitations
             * @default [
             *       "Bounded pilot-city coverage only; no Türkiye-wide coverage claim.",
             *       "Raw cosine similarity is uncalibrated and is not a probability or confidence.",
             *       "Reference proximity is not geographic proof; abstention remains valid.",
             *       "Private technical demonstration only; not cleared for public or production use."
             *     ]
             */
            limitations: string[];
        };
        /** MapillaryDemoStatusResponse */
        MapillaryDemoStatusResponse: {
            /**
             * State
             * @enum {string}
             */
            state: "disabled" | "not_ready" | "active";
            /** Enabled */
            enabled: boolean;
            /** Available */
            available: boolean;
            /** Reason Code */
            reason_code?: string | null;
            /** City */
            city?: string | null;
            /** Image Count */
            image_count: number;
            /**
             * Model Id
             * @default gberton/MegaLoc
             * @constant
             */
            model_id: "gberton/MegaLoc";
            /** Model Version */
            model_version?: string | null;
            /** Model Artifact Sha256 */
            model_artifact_sha256?: string | null;
            /**
             * Descriptor Dimension
             * @default 8448
             * @constant
             */
            descriptor_dimension: 8448;
            /** Index Version */
            index_version?: string | null;
            /** Index Checksum */
            index_checksum?: string | null;
            /**
             * Attribution Url
             * @default https://www.mapillary.com/
             * @constant
             */
            attribution_url: "https://www.mapillary.com/";
            /**
             * License Identifier
             * @default CC-BY-SA-4.0
             * @constant
             */
            license_identifier: "CC-BY-SA-4.0";
            /**
             * License Url
             * @default https://creativecommons.org/licenses/by-sa/4.0/
             * @constant
             */
            license_url: "https://creativecommons.org/licenses/by-sa/4.0/";
            /**
             * Experimental Status
             * @default private_technical_demo_not_production
             * @constant
             */
            experimental_status: "private_technical_demo_not_production";
            /**
             * Coverage Status
             * @default limited_pilot
             * @constant
             */
            coverage_status: "limited_pilot";
            /**
             * Coverage Label
             * @default Ankara reference pilot
             * @constant
             */
            coverage_label: "Ankara reference pilot";
            /**
             * Retrieval Provider
             * @default megaloc_mapillary_faiss
             * @constant
             */
            retrieval_provider: "megaloc_mapillary_faiss";
            /**
             * Retrieval Scope
             * @default ankara_reference_collection
             * @constant
             */
            retrieval_scope: "ankara_reference_collection";
            /**
             * Result Semantics
             * @default ankara_reference_collection_visual_similarity_not_general_geolocation
             * @constant
             */
            result_semantics: "ankara_reference_collection_visual_similarity_not_general_geolocation";
            /**
             * Similarity Semantics
             * @default cosine_similarity_not_confidence
             * @constant
             */
            similarity_semantics: "cosine_similarity_not_confidence";
            /**
             * Supported Region
             * @default Ankara pilot collection only
             * @constant
             */
            supported_region: "Ankara pilot collection only";
            /**
             * Evidence Version
             * @default phase3b3-mapillary-ankara-pilot-v1
             * @constant
             */
            evidence_version: "phase3b3-mapillary-ankara-pilot-v1";
            /**
             * Benchmark Version
             * @default phase3b3-mapillary-benchmark-v1
             * @constant
             */
            benchmark_version: "phase3b3-mapillary-benchmark-v1";
            /**
             * Limitations
             * @default [
             *       "Bounded pilot-city coverage only; no Türkiye-wide coverage claim.",
             *       "Raw cosine similarity is uncalibrated and is not a probability or confidence.",
             *       "Reference proximity is not geographic proof; abstention remains valid.",
             *       "Private technical demonstration only; not cleared for public or production use."
             *     ]
             */
            limitations: string[];
        };
    };
    responses: {
        /** @description Safe error response without an internal stack trace. */
        ProblemResponse: {
            headers: {
                [name: string]: unknown;
            };
            content: {
                "application/problem+json": components["schemas"]["ProblemDetails"];
            };
        };
    };
    parameters: {
        CaseId: string;
        HypothesisId: string;
        AnalysisId: string;
    };
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    getHealth: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Process is alive. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthResponse"];
                };
            };
        };
    };
    getReadiness: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Required local dependencies are ready. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ReadinessResponse"];
                };
            };
            /** @description A required local dependency is unavailable. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ReadinessResponse"];
                };
            };
        };
    };
    getCapabilities: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Current capability state; optional providers may be unavailable. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CapabilitiesResponse"];
                };
            };
        };
    };
    listProviders: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Provider status without secrets or local paths. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProviderStatusResponse"];
                };
            };
        };
    };
    listModels: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Safe model registry status. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelStatusResponse"];
                };
            };
            404: components["responses"]["ProblemResponse"];
        };
    };
    listAnalyses: {
        parameters: {
            query?: {
                limit?: number;
                offset?: number;
                status?: components["schemas"]["AnalysisStatus"];
                classification?: "real" | "simulated";
                provider?: string;
                search?: string;
                created_from?: string;
                created_to?: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Current TTL-bound history summaries. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AnalysisHistoryPage"];
                };
            };
            422: components["responses"]["ProblemResponse"];
        };
    };
    createAnalysis: {
        parameters: {
            query?: never;
            header?: {
                "Idempotency-Key"?: string;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": {
                    /** Format: binary */
                    image: string;
                    analysis_mode: components["schemas"]["AnalysisMode"];
                    cloud_processing_consent: boolean;
                    authorization_acknowledged: boolean;
                    /** @default false */
                    allow_cloud_assist?: boolean;
                };
            };
        };
        responses: {
            /** @description Analysis accepted. */
            202: {
                headers: {
                    Location?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AnalysisAccepted"];
                };
            };
            400: components["responses"]["ProblemResponse"];
            413: components["responses"]["ProblemResponse"];
            415: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            429: components["responses"]["ProblemResponse"];
        };
    };
    rerunAnalysis: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                analysis_id: components["parameters"]["AnalysisId"];
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Rerun accepted as a new analysis. */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AnalysisAccepted"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
        };
    };
    getAnalysis: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                analysis_id: components["parameters"]["AnalysisId"];
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Current analysis state. */
            200: {
                headers: {
                    "Cache-Control"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Analysis"];
                };
            };
            404: components["responses"]["ProblemResponse"];
        };
    };
    deleteAnalysis: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                analysis_id: components["parameters"]["AnalysisId"];
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Deletion is idempotently acknowledged. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DeleteResponse"];
                };
            };
            400: components["responses"]["ProblemResponse"];
        };
    };
    streamAnalysisEvents: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                analysis_id: components["parameters"]["AnalysisId"];
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description SSE stream. Event names are progress, heartbeat, completed, failed, and deleted. Each data payload validates as AnalysisEvent. */
            200: {
                headers: {
                    "Cache-Control"?: string;
                    "X-Accel-Buffering"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "text/event-stream": string;
                };
            };
            404: components["responses"]["ProblemResponse"];
            429: components["responses"]["ProblemResponse"];
        };
    };
    listEvaluations: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Available evaluation reports; simulated providers are excluded. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["EvaluationReportList"];
                };
            };
            404: components["responses"]["ProblemResponse"];
        };
    };
    getEvaluation: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                report_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Safe evaluation report summary. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["EvaluationReportSummary"];
                };
            };
            404: components["responses"]["ProblemResponse"];
        };
    };
    listDatasetQaReports: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Safe dataset-QA report summaries. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DatasetQAReportList"];
                };
            };
            404: components["responses"]["ProblemResponse"];
        };
    };
    getDatasetQaReport: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                report_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Structured QA issues without source paths or exact GPS. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DatasetQAReport"];
                };
            };
            404: components["responses"]["ProblemResponse"];
        };
    };
    getSystemIntelligence: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Aggregate operator-only model, pipeline, leakage-audit, and reference-index status without paths or secrets. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SystemIntelligenceResponse"];
                };
            };
            404: components["responses"]["ProblemResponse"];
        };
    };
    getMapillaryDemoStatus: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Credential-free disabled, not-ready, or active demo state. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MapillaryDemoStatusResponse"];
                };
            };
            403: components["responses"]["ProblemResponse"];
        };
    };
    queryMapillaryDemo: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_queryMapillaryDemo"];
            };
        };
        responses: {
            /** @description Attributed uncalibrated candidates or an explicit abstention. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MapillaryDemoQueryResponse"];
                };
            };
            403: components["responses"]["ProblemResponse"];
            413: components["responses"]["ProblemResponse"];
            415: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
        };
    };
    listCases: {
        parameters: {
            query?: {
                limit?: number;
                offset?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CasePage"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    createCase: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CaseCreateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CaseView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    getCase: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CaseView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    patchCase: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CasePatchRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CaseView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    listCaseMedia: {
        parameters: {
            query?: {
                limit?: number;
                offset?: number;
            };
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CaseMediaPage"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    createCaseMedia: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CaseMediaCreateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CaseMediaView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    linkCaseAnalysis: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
                analysis_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AnalysisLinkRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CaseMediaView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    materializeCaseEvidence: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
                analysis_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["MaterializeEvidenceRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MaterializationView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    listCaseEvidence: {
        parameters: {
            query?: {
                media_id?: string | null;
                analysis_id?: string | null;
                limit?: number;
                offset?: number;
            };
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["EvidencePage"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    listCaseHypotheses: {
        parameters: {
            query?: {
                media_id?: string | null;
                analysis_id?: string | null;
                limit?: number;
                offset?: number;
            };
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HypothesisPage"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    createHypothesisAdjudication: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
                hypothesis_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["AdjudicationCreateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AdjudicationView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    createOperatorHypothesis: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["OperatorHypothesisCreateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OperatorCorrectionView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    listCaseAuditEvents: {
        parameters: {
            query?: {
                limit?: number;
                offset?: number;
            };
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AuditEventPage"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
    getCaseAuditIntegrity: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                case_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AuditIntegrityView"];
                };
            };
            404: components["responses"]["ProblemResponse"];
            409: components["responses"]["ProblemResponse"];
            422: components["responses"]["ProblemResponse"];
            503: components["responses"]["ProblemResponse"];
        };
    };
}
