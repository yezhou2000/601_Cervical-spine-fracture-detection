import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot([0.69742, 0.67738, 0.66141, 0.67226, 0.66036, 0.65579], c='C0', label='loss')
plt.plot([0.70021, 0.69980, 0.69967, 0.70099, 0.70287, 0.70241], c='C1', label='val_loss')
plt.title('Competition metric')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()