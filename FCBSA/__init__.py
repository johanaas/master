from .fcbsa import FCBSA
from .dynFCBSA import dynFCBSA

def run_fcbsa(experiment, sample, model, params):
  if experiment == "fcbsa":
    return FCBSA(sample, model, params)
  elif experiment == "dyn-fcbsa0.5":
    return dynFCBSA(sample, model, params, 0.5)
  elif experiment == "dyn-fcbsa1":
    return dynFCBSA(sample, model, params, 1)
  elif experiment == "dyn-fcbsa1.5":
    return dynFCBSA(sample, model, params, 1.5)
  elif experiment == "dyn-fcbsa2":
    return dynFCBSA(sample, model, params, 2)
  elif experiment == "dyn-fcbsa2.5":
    return dynFCBSA(sample, model, params, 2.5)
  elif experiment == "dyn-fcbsa3":
    return dynFCBSA(sample, model, params, 3)
  else:
    raise ValueError("No init method with the name:", experiment)