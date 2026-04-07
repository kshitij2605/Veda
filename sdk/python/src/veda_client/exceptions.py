class VedaError(Exception):
    def __init__(self, message, status_code=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class CollectionNotFoundError(VedaError):
    pass


class ModalityNotSupportedError(VedaError):
    pass


class ValidationError(VedaError):
    pass
