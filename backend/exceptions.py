from rest_framework.exceptions import APIException
from rest_framework import status

class InvalidImageError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Invalid image file provided'
    default_code = 'invalid_image'

class ProcessingError(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = 'Error processing image'
    default_code = 'processing_error'

class ModelLoadingError(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = 'Model could not be loaded'
    default_code = 'model_loading_error'