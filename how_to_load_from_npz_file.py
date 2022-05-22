import numpy as np

np.set_printoptions(threshold=np.inf)

"""
The .npz file format is a way to store a bunch of numpy arrays. I ran 100 simulations and 
stored the data in the file 'data.npz'.
"""

all_data = np.load('data.npz')
file_names = all_data.files
print(file_names)

"""
To retrieve the first simulation data, we would use the following:
"""

first_file_name = file_names[0]
first_simulation_data = all_data[first_file_name]
print(first_simulation_data)
