import torch

from PIL import Image
from typing import Union, List

from utils import log_classification

def predict(model, transform, images: Union[str, List[str]], params: dict):
    def single_prediction(img, transform, results_list):
        tens = transform(Image.open(img))
        tens = tens.unsqueeze(0)

        with torch.no_grad():
            prediction = model(tens)
            
        _, predicted = torch.max(prediction.data, 1)
        
        if model.log:
            return log_classification(img, predicted.numpy(), params, results_list)
    
    assert isinstance(images, str) or isinstance(images, list), TypeError(f'Input should be str or list of str; it was {type(images)}')
    
    if params['device'] == 'gpu':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else: 
        torch.device('cpu')

    model.load_state_dict(torch.load(f'models/model_{params["model_class"]}.pickle', map_location = device))
    model.eval()

    results = []

    if type(images) is str:
        return single_prediction(images, transform, results)
    
    elif type(images) is list:      
        for img in images:
            results = single_prediction(img, transform, results)
            
        return results