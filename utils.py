import json
import torchvision

from datetime import datetime
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def log_detection(path, pred, results_dict):
    result = {}
    result['image'] = path
    result['scores'] = pred['scores'].detach().numpy().tolist()
    result['boxes'] = pred['boxes'].detach().numpy().tolist()
    #result['masks'] = pred['masks'].detach().numpy()
    results_dict.append(result)
    
    return results_dict

def log_classification(image_path, pred, params, results):
    uniques = params['uniques']
    result = {}
    result['image'] = image_path
    result[params['model_class']] = uniques[int(pred)]

    results.append(result)

    return results

def get_model_instance_detection(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_model_instance_classification():
    transform_val = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Resize((512, 512)),
         torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                          std = [0.229, 0.224, 0.225])
         ])        
        
    return torchvision.models.alexnet(pretrained=True), transform_val

def save_log(results, params):
    ### Metodo de salvar os resultados
    now = datetime.now().strftime("%m%d%Y-%H%M%S")
    log_file = f'logs/log_results_{params["model_class"]}_{now}.json'

    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)
        