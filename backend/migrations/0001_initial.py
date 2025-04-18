# Generated by Django 4.2.7 on 2025-04-18 22:59

import backend.models
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ClassificationStatistics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(default=django.utils.timezone.now, unique=True)),
                ('total_classifications', models.IntegerField(default=0)),
                ('cat_count', models.IntegerField(default=0)),
                ('dog_count', models.IntegerField(default=0)),
                ('unknown_count', models.IntegerField(default=0)),
                ('avg_confidence', models.FloatField(default=0.0)),
                ('avg_processing_time', models.FloatField(default=0.0)),
            ],
            options={
                'verbose_name': 'Classification Statistics',
                'verbose_name_plural': 'Classification Statistics',
                'ordering': ['-date'],
            },
        ),
        migrations.CreateModel(
            name='UploadedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to=backend.models.upload_path_handler)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('processed', models.BooleanField(default=False)),
                ('ip_address', models.GenericIPAddressField(blank=True, null=True)),
                ('user_agent', models.TextField(blank=True, null=True)),
            ],
            options={
                'verbose_name': 'Uploaded Image',
                'verbose_name_plural': 'Uploaded Images',
                'ordering': ['-uploaded_at'],
            },
        ),
        migrations.CreateModel(
            name='ClassificationResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pet_type', models.CharField(choices=[('cat', 'Cat'), ('dog', 'Dog'), ('unknown', 'Unknown')], default='unknown', max_length=10)),
                ('confidence', models.FloatField(default=0.0)),
                ('processing_time', models.FloatField(default=0.0)),
                ('model_version', models.CharField(default='v1.0', max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('additional_data', models.JSONField(blank=True, null=True)),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='classification_results', to='backend.uploadedimage')),
            ],
            options={
                'verbose_name': 'Classification Result',
                'verbose_name_plural': 'Classification Results',
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['pet_type'], name='backend_cla_pet_typ_5170f2_idx'), models.Index(fields=['confidence'], name='backend_cla_confide_e7954a_idx')],
            },
        ),
    ]
