from django import forms
from django.core.validators import FileExtensionValidator

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Select an image',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*',
            'id': 'imageInput'
        }),
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])],
        help_text='Supported formats: JPG, JPEG, PNG (Max 5MB)'
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            if image.size > 5 * 1024 * 1024:  # 5MB limit
                raise forms.ValidationError("Image file too large (maximum 5MB)")
        return image
    

    