import glob2
from predictors import *

def main():
    images = glob2.glob('images/*')
    
    predict = DetectMilha()
    #predict = ClassifyModelo()
    
    predict.call(images)
    predict.print_results()
    predict.save_results()
    
if __name__ == '__main__':
    main()