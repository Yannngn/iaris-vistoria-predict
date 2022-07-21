import pandas as pd

from argparse import ArgumentParser

from predictors import *

def main(hparams):
    # match hparams.model_class:
    #     case 'farol':
    #         predict = DetectFarol()
    #     case 'lampada_tras':
    #         predict = DetectLampadaTras()
    #     case 'parabrisa':
    #         predict = DetectParabrisa()
    #     case 'roda':
    #         predict = DetectRoda()
    #     case _:
    #         raise('valor inválido para --model_class')
 
    if hparams.model_class == 'farol':
        predict = DetectFarol()
    elif hparams.model_class == 'lampada_tras':
        predict = DetectLampadaTras()
    elif hparams.model_class == 'parabrisa':
            predict = DetectParabrisa()
    elif hparams.model_class == 'roda':
        predict = DetectRoda()
    elif hparams.model_class == 'cor':
            predict = ClassifyCor()
    elif hparams.model_class == 'modelo':
        predict = ClassifyModelo()
    else:
        raise('valor inválido para --model_class') 
    
    #predict = ClassifyModelo()
    predict.params['in_path'] = hparams.in_path
    predict.params['out_path'] = hparams.out_path
    predict.params['model_path'] = hparams.model_path

    df = pd.read_csv(hparams.in_path)
    images = df.image_path.tolist()
   
    if hparams.model_type == 'classification':
        predict.params['uniques'] = df.cor_label.unique().tolist()
        predict.params['num_classes'] = len(df.cor_label.unique().tolist())
        predict.params['layer_2_size'] = 4096
        predict.params['layer_4_size'] = 1024

    predict.call(images)
    predict.print_results()
    predict.save_results()

class Args:  
    @staticmethod
    def add_args(parent_parser: ArgumentParser) -> None:
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--model_class', '-c', type=str, help="model name ['farol', 'roda', 'lampada_tras', 'parabrisa', 'cor', 'modelo']")
        parser.add_argument('--model_type', '-t', type=str, help="model type ['classification', 'detection']")
        parser.add_argument('--in_path', '-i', type=str, help="path of images")
        parser.add_argument('--out_path', '-o', default=None, type=str, help="path of output for the masks")
        parser.add_argument('--model_path', '-m', type=str, help="path of the model weights")
        
        return parser
 
if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)

    parser = Args.add_args(parent_parser)
    hparams = parser.parse_args()
    
    main(hparams)