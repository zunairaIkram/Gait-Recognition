import pickle

# load data from already make pickle file
def load_data_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
