from django.db import models
import uuid
import os
from django.utils import timezone
from django.conf import settings


def upload_path_handler(instance, filename):
    """Generate a unique path for uploaded images."""
    # Get the file extension
    ext = filename.split('.')[-1]
    # Generate a unique filename using UUID
    filename = f"{uuid.uuid4().hex}.{ext}"
    # Return the full path
    return os.path.join('uploads', filename)


class UploadedImage(models.Model):
    """Model for storing uploaded images."""
    
    image = models.ImageField(upload_to=upload_path_handler)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Uploaded Image'
        verbose_name_plural = 'Uploaded Images'
    
    def __str__(self):
        return f"Image {self.id} uploaded at {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('image_detail', args=[str(self.id)])
    
    @property
    def filename(self):
        return os.path.basename(self.image.name)
    
    @property
    def classification_result(self):
        """Return the latest classification result."""
        try:
            return self.classification_results.latest('created_at')
        except ClassificationResult.DoesNotExist:
            return None


class ClassificationResult(models.Model):
    """Model for storing classification results."""
    
    # Define pet types as choices
    CAT = 'cat'
    DOG = 'dog'
    UNKNOWN = 'unknown'
    
    PET_TYPE_CHOICES = [
        (CAT, 'Cat'),
        (DOG, 'Dog'),
        (UNKNOWN, 'Unknown'),
    ]
    
    # Relationship to the uploaded image
    image = models.ForeignKey(
        UploadedImage, 
        on_delete=models.CASCADE, 
        related_name='classification_results'
    )
    
    # Classification details
    pet_type = models.CharField(
        max_length=10, 
        choices=PET_TYPE_CHOICES, 
        default=UNKNOWN
    )
    confidence = models.FloatField(default=0.0)  # Confidence score (0.0 to 1.0)
    processing_time = models.FloatField(default=0.0)  # Processing time in seconds
    
    # Classification metadata
    model_version = models.CharField(max_length=50, default='v1.0')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Additional result data (JSON field for flexibility)
    additional_data = models.JSONField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Classification Result'
        verbose_name_plural = 'Classification Results'
        indexes = [
            models.Index(fields=['pet_type']),
            models.Index(fields=['confidence']),
        ]
    
    def __str__(self):
        return f"{self.get_pet_type_display()} ({self.confidence:.2f}) - Image {self.image.id}"
    
    @property
    def is_confident(self):
        """Return True if the confidence score is above the threshold."""
        CONFIDENCE_THRESHOLD = 0.7  # 70% confidence
        return self.confidence >= CONFIDENCE_THRESHOLD
    
    @property
    def confidence_percentage(self):
        """Return the confidence as a percentage."""
        return f"{self.confidence * 100:.1f}%"
    
    @property
    def created_at_formatted(self):
        """Return a formatted timestamp."""
        return self.created_at.strftime("%Y-%m-%d %H:%M:%S")


class ClassificationStatistics(models.Model):
    """Model for storing system-wide classification statistics."""
    
    date = models.DateField(default=timezone.now, unique=True)
    total_classifications = models.IntegerField(default=0)
    cat_count = models.IntegerField(default=0)
    dog_count = models.IntegerField(default=0)
    unknown_count = models.IntegerField(default=0)
    avg_confidence = models.FloatField(default=0.0)
    avg_processing_time = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-date']
        verbose_name = 'Classification Statistics'
        verbose_name_plural = 'Classification Statistics'
    
    def __str__(self):
        return f"Stats for {self.date}: {self.total_classifications} classifications"
    
    @classmethod
    def update_statistics(cls, classification_result):
        """Update statistics based on a new classification result."""
        today = timezone.now().date()
        stats, created = cls.objects.get_or_create(date=today)
        
        # Update counts
        stats.total_classifications += 1
        if classification_result.pet_type == ClassificationResult.CAT:
            stats.cat_count += 1
        elif classification_result.pet_type == ClassificationResult.DOG:
            stats.dog_count += 1
        else:
            stats.unknown_count += 1
        
        # Update averages
        all_results = ClassificationResult.objects.filter(
            created_at__date=today
        )
        stats.avg_confidence = all_results.aggregate(models.Avg('confidence'))['confidence__avg'] or 0.0
        stats.avg_processing_time = all_results.aggregate(models.Avg('processing_time'))['processing_time__avg'] or 0.0
        
        stats.save()
        return stats
    

