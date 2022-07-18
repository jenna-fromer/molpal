import os 
from molpal.models.nnmodels import NNEnsembleModel, NNModel
from molpal import featurizer
from molpal.featurizer import featurize
from rdkit import Chem
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np 
import ray 

# tested modules: 

if not ray.is_initialized(): ray.init(num_cpus=12, num_gpus=1)

featurizer = featurizer.Featurizer()

print(os.getcwd())
nnmodel = NNEnsembleModel(
    input_size = 2048,
    test_batch_size=100,
    ensemble_size=3,
    dropout=0)

smis_path = 'data/Enamine10k_scores.csv.gz'
df = pd.read_csv(smis_path)
xs = list(df['smiles'])
fps = [featurizer(smi) for smi in xs]

# generate ys 
transformer = np.random.rand(2048)

ys = np.array([np.matmul(fp,transformer) for fp in fps])

nnmodel.train(xs=xs, 
        ys=ys ,
        featurizer=featurizer,
        retrain=False,
        epochs=50)

preds = nnmodel.get_means(xs)
means, vars = nnmodel.get_means_and_vars(xs)


# parity plot 
fig, ax = plt.subplots()
ax.scatter(preds,ys)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Ensemble of NNs (size=3)') 
ax.plot([np.min(ys),np.max(ys)],[np.min(ys),np.max(ys)],'--k' )
plt.savefig('NNEnsemble.jpg')


print(ys)