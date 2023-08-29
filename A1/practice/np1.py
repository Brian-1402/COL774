import numpy as np

l1 = np.array([1, 2, 3, 4])
l2 = np.array(
    [
        [1, 2, 3, 4],
        [24, 22, 23, 21],
    ]
)
l3 = np.array([1, 2])


# print(l2.shape)
# print(l2)
# print(np.sort(l2, axis=0))
# print(np.divide(l2, l1))
# print(l2 / l1)
# print(l2.T)
print((l2 @ l1).shape)
