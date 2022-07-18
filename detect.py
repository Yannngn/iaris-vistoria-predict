import numpy as np
import torch
import torchvision

from PIL import Image
from typing import Union, List

from utils import log_detection

def predict(model, images: Union[str, List[str]], params: dict): 
    def single_prediction(img, results_list):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Resize((320, 320))])
        
        tens = transform(Image.open(img).convert('RGB'))
        tens = tens.unsqueeze(0)

        with torch.no_grad():
            prediction = model(tens)
            pred = prediction[0]
            
            # pred['masks'][pred['masks'] > params['min_score']] = 1
            # pred['masks'] = pred['masks'].type(torch.uint8)

            for j, score in enumerate(pred['scores']):
                if score >= params['min_score']:
                    mask = torch.from_numpy(np.array(pred['masks'][j])).type(torch.float32)
                    torchvision.utils.save_image(mask, f'{params["out_path"]}/CAR{str(params["count"]).zfill(5)}_{params["model_class"]}_{j}.png')

            params["count"] += 1

            return log_detection(img, prediction[0], results_list, params)

    assert isinstance(images, str) or isinstance(images, list), TypeError(f'Input should be str or list of str; it was {type(images)}')

    if params['device'] == 'gpu':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else: 
        torch.device('cpu')

    model.load_state_dict(torch.load(params['model_path'], map_location = device))
    model.eval()

    results = []

    if type(images) is str:
        return single_prediction(images, results)
    
    elif type(images) is list:      
        for img in images:
            results = single_prediction(img, results)
            
        return results