import numpy as np

kernel_x = np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])

for k in range(0,3):
        for u in range(0,3):
            print(kernel_x[k][u])