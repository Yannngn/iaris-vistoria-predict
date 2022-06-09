import torch
import torchvision

from PIL import Image
from typing import Union, List

from utils import log_detection

def predict(model, images: Union[str, List[str]], params: dict): 
    def single_prediction(img, results_list):
        transform = torchvision.transforms.ToTensor()
        tens = transform(Image.open(img))
        tens = tens.unsqueeze(0)

        with torch.no_grad():
            prediction = model(tens)

            return log_detection(img, prediction[0], results_list)

    assert isinstance(images, str) or isinstance(images, list), TypeError(f'Input should be str or list of str; it was {type(images)}')

    if params['device'] == 'gpu':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else: 
        torch.device('cpu')

    model.load_state_dict(torch.load(f'models/model_{params["model_class"]}.pickle', map_location = device))
    model.eval()

    results = []

    if type(images) is str:
        return single_prediction(images, results)
    
    elif type(images) is list:      
        for img in images:
            results = single_prediction(img, results)
            
        return results