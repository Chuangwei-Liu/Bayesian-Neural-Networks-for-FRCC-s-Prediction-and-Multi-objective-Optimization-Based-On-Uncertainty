import dill

# Define the function to save the pkl file
def save_file(file, obj):
    with open(file, 'wb') as f:
        dill.dump(obj, f)

# Define the function to load the pkl file
def load_file(file):
    with open(file, 'rb') as f:
        return dill.load(f)