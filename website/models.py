from django.db import models

# Create your models here.
from django.db import models
import uuid

class Image(models.Model):
    unique_identifier = models.UUIDField(default=uuid.uuid4, unique=True)
    image_data = models.BinaryField()