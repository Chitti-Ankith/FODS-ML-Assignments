import matplotlib.pyplot as plt 

x = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

y = [21.766154670409264, 28.787390440164643, 46.104738244495,109.4398763042918, 510.3246043611067, 1767.3136542233121, 5829.575574869379]

plt.plot(x,y)
plt.xlabel('Regularisation Coefficient') 

plt.ylabel('Loss') 
plt.show() 