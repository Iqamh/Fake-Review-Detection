import pickle
import numpy as np


result_pickle_file = "model_result.pkl"
# model_pickle_file = "model_data.pkl"

# with open(model_pickle_file, "rb") as file:
#     models = pickle.load(file)

# mlp_80_10_10 = models['mlp_80_10_10']
# mlp_70_20_10 = models['mlp_70_20_10']
# mlp_60_30_10 = models['mlp_60_30_10']

# print(mlp_80_10_10)

with open(result_pickle_file, "rb") as file:
    result = pickle.load(file)

result_80_10_10 = result["80_10_10"]
result_70_20_10 = result["70_20_10"]
result_60_30_10 = result["60_30_10"]

# 90-10
print(result_80_10_10["classification_report"])
print("Avg Loss: ", f'{np.mean(result_80_10_10["loss"]):.3f}')
print("Avg Val Loss: ", f'{np.mean(result_80_10_10["loss_validation"]):.3f}')
print("Avg Accuracy: ", f'{np.mean(result_80_10_10["accuracy_train"]):.3f}')
print("Avg Val Accuracy: ",
      f'{np.mean(result_80_10_10["accuracy_validation"]):.3f}')
print("\n")

# 80-20
print(result_70_20_10["classification_report"])
print("Avg Loss: ", f'{np.mean(result_70_20_10["loss"]):.3f}')
print("Avg Val Loss: ", f'{np.mean(result_70_20_10["loss_validation"]):.3f}')
print("Avg Accuracy: ", f'{np.mean(result_70_20_10["accuracy_train"]):.3f}')
print("Avg Val Accuracy: ",
      f'{np.mean(result_70_20_10["accuracy_validation"]):.3f}')
print("\n")

# 70-30
print(result_60_30_10["classification_report"])
print("Avg Loss: ", f'{np.mean(result_60_30_10["loss"]):.3f}')
print("Avg Val Loss: ", f'{np.mean(result_60_30_10["loss_validation"]):.3f}')
print("Avg Accuracy: ", f'{np.mean(result_60_30_10["accuracy_train"]):.3f}')
print("Avg Val Accuracy: ",
      f'{np.mean(result_60_30_10["accuracy_validation"]):.3f}')
