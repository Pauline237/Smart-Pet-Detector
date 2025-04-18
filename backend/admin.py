from django.contrib import admin
from django.utils.html import format_html
from .models import UploadedImage, ClassificationResult, ClassificationStatistics


class ClassificationResultInline(admin.TabularInline):
    """Inline admin for ClassificationResult."""
    model = ClassificationResult
    readonly_fields = ('pet_type', 'confidence', 'processing_time', 'model_version', 'created_at')
    extra = 0
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    """Admin for UploadedImage model."""
    list_display = ('id', 'thumbnail', 'uploaded_at', 'processed', 'ip_address', 'classification_type')
    list_filter = ('processed', 'uploaded_at')
    search_fields = ('id', 'ip_address')
    readonly_fields = ('image_preview',)
    inlines = [ClassificationResultInline]
    
    def thumbnail(self, obj):
        """Display a thumbnail of the image."""
        if obj.image:
            return format_html('<img src="{}" width="50" height="50" style="object-fit: cover;" />', obj.image.url)
        return "-"
    thumbnail.short_description = 'Thumbnail'
    
    def image_preview(self, obj):
        """Display a larger preview of the image."""
        if obj.image:
            return format_html('<img src="{}" width="300" style="max-height: 300px; object-fit: contain;" />', obj.image.url)
        return "-"
    image_preview.short_description = 'Image Preview'
    
    def classification_type(self, obj):
        """Display the classification result with a colored badge."""
        result = obj.classification_result
        if not result:
            return format_html('<span style="color: gray;">Not processed</span>')
        
        colors = {
            'cat': 'blue',
            'dog': 'green',
            'unknown': 'orange'
        }
        color = colors.get(result.pet_type, 'gray')
        
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 10px;">{} ({})</span>',
            color,
            result.get_pet_type_display(),
            result.confidence_percentage
        )
    classification_type.short_description = 'Classification'


@admin.register(ClassificationResult)
class ClassificationResultAdmin(admin.ModelAdmin):
    """Admin for ClassificationResult model."""
    list_display = ('id', 'pet_type', 'confidence_percentage', 'processing_time', 'created_at_formatted')
    list_filter = ('pet_type', 'created_at', 'model_version')
    search_fields = ('id', 'pet_type')
    readonly_fields = ('image_link',)
    
    def image_link(self, obj):
        """Display a link to the related image."""
        if obj.image:
            return format_html('<a href="{}">{}</a>', obj.image.get_absolute_url(), obj.image)
        return "-"
    image_link.short_description = 'Image'


@admin.register(ClassificationStatistics)
class ClassificationStatisticsAdmin(admin.ModelAdmin):
    """Admin for ClassificationStatistics model."""
    list_display = ('date', 'total_classifications', 'cat_count', 'dog_count', 
                   'unknown_count', 'avg_confidence_display', 'avg_processing_time_display')
    readonly_fields = ('date', 'total_classifications', 'cat_count', 'dog_count', 
                       'unknown_count', 'avg_confidence', 'avg_processing_time')
    
    def avg_confidence_display(self, obj):
        """Display average confidence as a percentage."""
        return f"{obj.avg_confidence * 100:.1f}%"
    avg_confidence_display.short_description = 'Avg. Confidence'
    
    def avg_processing_time_display(self, obj):
        """Display average processing time in ms."""
        return f"{obj.avg_processing_time * 1000:.0f} ms"
    avg_processing_time_display.short_description = 'Avg. Processing Time'
    
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return False