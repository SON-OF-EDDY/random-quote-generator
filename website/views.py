from django.shortcuts import render
from django.http import HttpResponse,JsonResponse,HttpResponseRedirect
from django.urls import reverse
import cv2
import numpy as np
import torch
import requests
from PIL import Image
import base64
import io
import json

# Create your views here.

def index(request):
    return render(request,'index.html',{})

def preprocess_custom_digit_image(image, target_size=(28, 28)):
    # Ensure the image is binary (0 for background, 255 for digit)
    _, img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Resize the image to the target size (e.g., 28x28) without cropping
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to the range [0, 1]
    normalized_img = resized_img.astype(np.float32) / 255.0

    # Add a batch dimension
    preprocessed_image = np.expand_dims(normalized_img, axis=0)

    return preprocessed_image

def process_image(request):
    # Access the JSON data from the request body
    try:
        data = json.loads(request.body)
        image_url = data.get('image_url')
        # Now, image_url contains the value of 'image_url' from the JSON data
        # You can use it as needed in your view
        #return JsonResponse({'image url':image_url})
    except json.JSONDecodeError as e:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON data','e':str(e)})

    # image_url = request.POST.get('data')
    #
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # load pytorch model
    from django.conf import settings
    import os
    model_checkpoint_path = os.path.join(settings.BASE_DIR, 'website', 'pytorch', 'model.pth')
    checkpoint = torch.load(model_checkpoint_path)

    network = Net()
    network.load_state_dict(checkpoint)
    network.eval()
    #
    # # Decode the Data URL to get image data

    ####################################################################################################################
    ####################################################################################################################
    try:
        _, encoded_data = image_url.split(',', 1)
        image_data = base64.b64decode(encoded_data)
        #return JsonResponse({'image data': image_data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': 'Invalid Data URL','e':str(e)})

    ####################################################################################################################
    ####################################################################################################################

    # try:
    #     _, encoded_data = image_url.split(',', 1)
    #     image_data = base64.b64decode(encoded_data)
    #
    #     # Encode the binary image data as base64 and convert it to a string
    #     image_data_base64 = base64.b64encode(image_data).decode('utf-8')
    #
    #     return JsonResponse({'image_data': image_data_base64})
    # except Exception as e:
    #     return JsonResponse({'status': 'error', 'message': 'Invalid Data URL', 'e': str(e)})

    #
    # Convert the image data to a PIL Image
    try:
        image_data = Image.open(io.BytesIO(image_data)).convert('L')  # 'L' mode for grayscale
        #return JsonResponse({'message': 'going well so far...'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': 'Failed to process the image'})
    #
    # # Preprocess the image using your custom function
    input_image = preprocess_custom_digit_image(np.array(image_data))  # Convert PIL Image to numpy array

    #
    # # Convert the numpy array to a PyTorch tensor
    input_image = torch.tensor(input_image, dtype=torch.float32)

    #
    network.eval()
    with torch.no_grad():
        output = network(input_image)
    #
    predicted_class = torch.argmax(output, dim=1).item()
    #
    return JsonResponse({'status': 'success', 'message': 'Image processing complete','prediction':predicted_class})
