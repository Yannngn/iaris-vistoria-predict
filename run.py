import pandas as pd

from argparse import ArgumentParser

from predictors import *

def main(hparams):
    log = False
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
        predict = DetectFarol(log)
    elif hparams.model_class == 'lampada_tras':
        predict = DetectLampadaTras(log)
    elif hparams.model_class == 'parabrisa':
            predict = DetectParabrisa(log)
    elif hparams.model_class == 'roda':
        predict = ClassifyCor(log)
    elif hparams.model_class == 'cor':
            predict = DetectParabrisa(log)
    elif hparams.model_class == 'modelo':
        predict = ClassifyModelo(log)
    else:
        raise('valor inválido para --model_class') 
    
    #predict = ClassifyModelo()
    
    predict.params['in_path'] = hparams.in_path
    predict.params['out_path'] = hparams.out_path
    predict.params['model_path'] = hparams.model_path

    df = pd.read_csv(hparams.in_path)
    images = df.ref_image.tolist()
    
    predict.call(images)
    predict.print_results()
    predict.save_results()

class Args:  
    @staticmethod
    def add_args(parent_parser: ArgumentParser) -> None:
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--in_path', '-i', type=str, help="path of images")
        parser.add_argument('--out_path', '-o', type=str, help="path of output for the masks")
        parser.add_argument('--model_path', '-m', type=str, help="path of the model weights")
        parser.add_argument('--model_class', '-c', type=str, help="model name ['farol', 'roda', 'lampada_tras', 'parabrisa', 'cor', 'modelo']")
        
        return parser
 
if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)

    parser = Args.add_args(parent_parser)
    hparams = parser.parse_args()
    
    main(hparams)