from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.http import JsonResponse

from .ml_model.prediction_service import get_prediction_service
from .models import UploadedImage, ClassificationResult
from .forms import ImageUploadForm

def home(request):
    return render(request, 'index.html')

@require_http_methods(["GET", "POST"])
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file temporarily
            image_file = request.FILES['image']
            file_name = default_storage.save(f'tmp/{image_file.name}', image_file)
            file_path = default_storage.path(file_name)
            
            # Get prediction
            service = get_prediction_service()
            try:
                result = service.predict_image_file(file_path)
                
                # Create model instances if user is authenticated
                if request.user.is_authenticated:
                    uploaded_image = UploadedImage.objects.create(
                        user=request.user,
                        image=image_file,
                        ip_address=get_client_ip(request),
                        user_agent=request.META.get('HTTP_USER_AGENT', '')
                    )
                    
                    ClassificationResult.objects.create(
                        image=uploaded_image,
                        pet_type=result['class_name'],
                        confidence=result['confidence'],
                        processing_time=result['processing_time'],
                        model_version=result['model_version'],
                        additional_data={
                            'opposite_class': 'dog' if result['class_name'] == 'cat' else 'cat',
                            'opposite_confidence': 100 - result['confidence']
                        }
                    )
                
                # Prepare result data for template
                result_data = {
                    'class_name': result['class_name'],
                    'confidence': result['confidence'] * 100,  # Convert to percentage
                    'processing_time': result['processing_time'],
                    'model_version': result['model_version'],
                    'opposite_class': 'dog' if result['class_name'] == 'cat' else 'cat',
                    'opposite_confidence': 100 - (result['confidence'] * 100),
                    'image_url': default_storage.url(file_name)
                }
                
                return render(request, 'result.html', {
                    'result': result_data,
                    'result_json': result_data,
                    'image_url': default_storage.url(file_name)
                })
                
            except Exception as e:
                messages.error(request, f"Error processing image: {str(e)}")
                return redirect('home')
            finally:
                # Clean up - remove the temporary file
                if not request.user.is_authenticated:
                    default_storage.delete(file_name)
        else:
            messages.error(request, "Invalid image file. Please try again.")
            return redirect('home')
    return redirect('home')

def save_result(request):
    if request.method == 'POST' and request.user.is_authenticated:
        try:
            # Parse JSON data from request
            data = json.loads(request.body)
            
            # Create the uploaded image record
            uploaded_image = UploadedImage.objects.create(
                user=request.user,
                image=data['image_url'],
                ip_address=get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')
            )
            
            # Create the classification result
            ClassificationResult.objects.create(
                image=uploaded_image,
                pet_type=data['result']['class_name'],
                confidence=data['result']['confidence'],
                processing_time=data['result']['processing_time'],
                model_version=data['result']['model_version'],
                additional_data={
                    'opposite_class': data['result']['opposite_class'],
                    'opposite_confidence': data['result']['opposite_confidence']
                }
            )
            
            return JsonResponse({'success': True})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip