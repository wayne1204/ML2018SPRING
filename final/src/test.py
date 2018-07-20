from utils import *
from para import *
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Sound classification')
parser.add_argument('--result_path', type=str)
parser.add_argument('--models', type=str)

args = parser.parse_args()

models = []
num_model = len(args.models.split(','))
models = (args.models.split(','))
predc = 0
for i in range(num_model):
    prob = np.load(models[i])
    predc  += prob**0.25
#predc = predc / float(num_model)
top_3 = np.array(category)[np.argsort(-predc, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]

data = pd.read_csv('../data/sample_submission.csv').values
for i, element in enumerate(predicted_labels):
    data[i][1] = element
columns = ['fname', 'label']
df = pd.DataFrame(data=data, columns=columns)
df.to_csv(args.result_path, encoding='utf-8', index=False)


