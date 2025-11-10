# %%
import numpy as np

class Test():
    def __init__(self):
        self.P = np.zeros((3,3))

# %%
t = Test()

P_stack = np.empty((2,3,3))
P_stack[0] = t.P
P_stack[1] = t.P + 1
print(P_stack)

# %%
P_stack[0][0][0] = 10

# %%
