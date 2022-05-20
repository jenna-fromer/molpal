import os 
from molpal.models.nnmodels import NNEnsembleModel
from molpal import featurizer
from rdkit import Chem
import numpy as np 
import ray 

if not ray.is_initialized(): ray.init(num_cpus=12, num_gpus=1)

featurizer = featurizer.Featurizer()

print(os.getcwd())
nnmodel = NNEnsembleModel(
    input_size = 2048,
    test_batch_size=100,
    dropout=0,
    ensemble_size=3)

xs = []
ys = np.random.randn(50,1)

testsmi = 'CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=C(N=CC(=C2)C3=CN(N=C3)C4CCNCC4)N'
mol = Chem.MolFromSmiles(testsmi)
for _ in range(50):
  smi = Chem.MolToSmiles(mol, doRandom=True)
  xs.append(smi)


model = nnmodel.train(xs=xs, 
        ys=ys ,
        featurizer=featurizer,
        retrain=False,
        epochs=200)

print(ys)