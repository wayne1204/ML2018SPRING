# FileName     [ mergr.py ]
# Date		   [ 2018.4]
# Synopsis     [ merge several model to one model ]

from keras import backend as K
from keras.models import load_model
from sys import argv
from keras.layers import Input, Dense, merge,average
from keras.models import Model


def main():
	models = [argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]]
	ymodel = []
	for i in models:
		ymodel.append(load_model(i))
	
	model_input = Input(shape=ymodel[0].input_shape[1:])
	
	yModels=[model(model_input) for model in ymodel]
	
	yAvg=average(yModels)
	
	modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
	
	modelEns.save('my_ensemble_model.h5')



if __name__ == '__main__':
    main()
