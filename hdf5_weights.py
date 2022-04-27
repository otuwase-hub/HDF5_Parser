import h5py
import numpy as np

def printStructure(fileName):
    layernames = []
    with h5py.File(fileName,mode='r') as f:
        for key in f:
            layernames.append(key)
            o = f[key]
            print(key,o)
            for key1 in o:
                p = o[key1]
                print(key1, p)
                for key2 in p:
                    print(key2, p[key2])
    return layernames

def is_dataset(obj):
    if isinstance(obj, h5py.Dataset):
        return True
    return False

def get_dataset(datasets, obj):
    if not is_dataset(obj):
        for key in obj:
            x = obj[key]
            get_dataset(datasets,x)
    else:
        datasets.append(obj)

def get_layer_weights(layername, filename):
    weights = []
    with h5py.File(filename,mode='r') as f:
        for key in f:
            if layername in key:
                obj = f[key]
                datasets = []
                get_dataset(datasets, obj)
                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
    return weights

layernames = printStructure('weights.best.hdf5')
print(layernames)
for layername in layernames:
    bias_weights = get_layer_weights(layername,'./weights.best.hdf5')
    # read bias
    if bias_weights:
        bias_textfile = open(f'{layername}_biases.txt', 'w')
        for bias in bias_weights[0]:
            bias_textfile.write(f'{bias}\n')
        bias_textfile.close()
        weight_textfile = open(f'{layername}_weights.txt', 'w')
        # read weights
        for weights in bias_weights[1]:
            for weight in weights:
                weight_textfile.write(f'{weight} ')
            weight_textfile.write('\n')
        weight_textfile.close()
        print(bias_weights[1].shape)
        print(bias_weights[0].shape)
