"""
references:
https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
"""

import torchvision.models as models
from torchvision import transforms, utils
from PIL import Image
import io
import requests

# Random cat img taken from Google
IMG_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg'
response = requests.get(IMG_URL)
img = Image.open(io.BytesIO(response.content))  # Read bytes and store as an img.
img.show()

transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

# resnet18 = models.resnet18(pretrained=True)
def convoluted_image(resnet18, x):
    x = resnet18.conv1(x)
    x = resnet18.bn1(x)
    x = resnet18.relu(x)
    x = resnet18.maxpool(x)
    
    x = resnet18.layer1(x)
    x = resnet18.layer2(x)
    x = resnet18.layer3(x)
    x = resnet18.layer4(x)
    
    x = resnet18.avgpool(x)
    
    # The folloiwng is the prediction part
    # x = x.view(x.size(0), -1)
    # x = resnet18.fc(x)