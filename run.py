import glob2

from argparse import ArgumentParser

from predictors import *

def main(hparams):
    images = glob2.glob(hparams.in_path+'*')
    
    predict = DetectMilha()
    #predict = ClassifyModelo()
    predict.params['in_path'] = hparams.in_path
    predict.params['out_path'] = hparams.out_path
    predict.call(images)
    predict.print_results()
    predict.save_results()

class Args:  
    @staticmethod
    def add_args(parent_parser: ArgumentParser) -> None:
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--in_path', '-i', type=str, help="path of images")
        parser.add_argument('--out_path', '-o', type=str, help="path of output for the masks")
        parser.add_argument('--model_path', '-o', type=str, help="path of the model weights")
        
        return parser
 
if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)

    parser = Args.add_args(parent_parser)
    hparams = parser.parse_args()
    
    main(hparams)