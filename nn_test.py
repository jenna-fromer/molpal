import os 
from molpal.models.nnmodels import NNModel
from molpal import featurizer
from rdkit import Chem
import numpy as np 
import ray 

if not ray.is_initialized(): ray.init()

featurizer = featurizer.Featurizer()

print(os.getcwd())
nnmodel = NNModel(
    input_size = 2048,
    test_batch_size=100,
    dropout=0,
    model_seed=1 )

xs = []
ys = np.random.randn(50,1)

testsmi = 'CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=C(N=CC(=C2)C3=CN(N=C3)C4CCNCC4)N'
mol = Chem.MolFromSmiles(testsmi)
for _ in range(50):
  smi = Chem.MolToSmiles(mol, doRandom=True)
  xs.append(smi)


nnmodel.train(xs=xs, 
        ys=ys ,
        featurizer=featurizer,
        retrain=False)