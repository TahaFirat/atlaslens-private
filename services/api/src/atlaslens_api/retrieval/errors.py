class RetrievalError(RuntimeError):
    pass


class ProviderUnavailableError(RetrievalError):
    pass


class ManifestValidationError(RetrievalError, ValueError):
    pass


class MetadataConflictError(RetrievalError):
    pass


class IndexIntegrityError(RetrievalError):
    pass


class IndexNotFoundError(RetrievalError):
    pass
