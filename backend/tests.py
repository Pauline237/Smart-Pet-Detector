from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
import os

class PetClassifierTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.upload_url = reverse('upload')
        self.test_image_path = os.path.join(settings.BASE_DIR, 'pet_classifier', 'tests', 'test_images')
        
    def test_home_page(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'index.html')
        self.assertContains(response, 'Upload Pet Image')
        
    def test_image_upload(self):
        # Test with valid image
        with open(os.path.join(self.test_image_path, 'test_cat.jpg'), 'rb') as img:
            response = self.client.post(self.upload_url, {'image': img})
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, 'result.html')
            self.assertContains(response, 'Our Analysis:')
            
        # Test with invalid file type
        with open(os.path.join(self.test_image_path, 'test.txt'), 'rb') as file:
            response = self.client.post(self.upload_url, {'image': file})
            self.assertEqual(response.status_code, 302)  # Redirects to home
            # Check for error message in redirected page
            response = self.client.get(reverse('home'))
            self.assertContains(response, 'Invalid image file')
            
    def test_save_result_authenticated(self):
        # Create a test user
        from django.contrib.auth.models import User
        user = User.objects.create_user(username='testuser', password='12345')
        self.client.login(username='testuser', password='12345')
        
        # Test saving result
        test_data = {
            'image_url': '/media/test.jpg',
            'result': {
                'class_name': 'cat',
                'confidence': 95.5,
                'processing_time': 1.2,
                'model_version': 'v1.0',
                'opposite_class': 'dog',
                'opposite_confidence': 4.5
            }
        }
        response = self.client.post(
            reverse('save_result'),
            data=test_data,
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        
        # Verify the record was created
        from .models import UploadedImage, ClassificationResult
        self.assertEqual(UploadedImage.objects.count(), 1)
        self.assertEqual(ClassificationResult.objects.count(), 1)
        
    def test_save_result_unauthenticated(self):
        test_data = {
            'image_url': '/media/test.jpg',
            'result': {
                'class_name': 'cat',
                'confidence': 95.5,
                'processing_time': 1.2,
                'model_version': 'v1.0',
                'opposite_class': 'dog',
                'opposite_confidence': 4.5
            }
        }
        response = self.client.post(
            reverse('save_result'),
            data=test_data,
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['success'])