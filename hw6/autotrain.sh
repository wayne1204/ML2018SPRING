python3 mf_train.py train --path model/mf_model_1.h5 --dim 256
python3 mf_train.py train --path model/mf_model_2.h5 --dim 128
python3 mf_train.py train --path model/mf_model_3.h5 --dim 64
python3 mf_train.py test --path model/mf_model_1.h5 --ans ana1.txt
python3 mf_train.py test --path model/mf_model_2.h5 --ans ans2.txt
python3 mf_train.py test --path model/mf_model_3.h5 --ans ans3.txt
python3 dnn_train.py --path model/dnn_model_1.h5 --dim 256
python3 dnn_train.py --path model/dnn_model_2.h5 --dim 128
python3 dnn_train.py --path model/dnn_model_3.h5 --dim 64
python3 dnn_test.py test --path model/dnn_model_1.h5 --ans ans1.csv
python3 dnn_test.py test --path model/dnn_model_2.h5 --ans ans2.csv
python3 dnn_test.py test --path model/dnn_model_3.h5 --ans ans3.csv
