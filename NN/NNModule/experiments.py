
    from kapre.composed import get_melspectrogram_layer
    import numpy as np
    from scipy.io.wavfile import read

    sr,data = read("anotherbrick.wav")
    data = data.reshape(-1,1)
input_shape = (int(16000*1.0), 1)
i = get_melspectrogram_layer(input_shape=input_shape,
                             pad_end=True,
                             sample_rate=16000,
                             return_decibel=True,
                             input_data_format='channels_last',
                             output_data_format='channels_last')
                            

print(i(data))



# 

#     X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
#     # Shuffle the training dataset
#     keys = np.array(range(X.shape[0]))
#     np.random.shuffle(keys)
#     X = X[keys]
#     y = y[keys]
#     X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# def load_mnist_dataset(dataset, path):
#     # Scan all the directories and create a list of labels
#     labels = os.listdir(os.path.join(path, dataset))
#     # Create lists for samples and labels
#     X = []
#     y = []
#     # For each label folder
#     for label in labels:
#         # And for each image in given folder
#         for file in os.listdir(os.path.join(path, dataset, label)):
#             # Read the image
#             image = cv2.imread(
#                 os.path.join(path, dataset, label, file),
#                 cv2.IMREAD_UNCHANGED)
#             # And append it and a label to the lists
#             X.append((image.reshape(1,28,28).astype(np.float32) - 127.5) / 127.5)
#             y.append(label)
#     # Convert the data to proper numpy arrays and return
#     return np.array(X), np.array(y).astype('uint8')
# # MNIST dataset (train + test)


# def create_data_mnist(path):
#     # Load both sets separately
#     X, y = load_mnist_dataset('train', path)
#     X_test, y_test = load_mnist_dataset('test', path)
#     # And return all the data
#     return X, y, X_test, y_test


# # Label index to label name relation
# fashion_mnist_labels = {
#     0: 'T-shirt/top',
#     1: 'Trouser',
#     2: 'Pullover',
#     3: 'Dress',
#     4: 'Coat',
#     5: 'Sandal',
#     6: 'Shirt',
#     7: 'Sneaker',
#     8: 'Bag',
#     9: 'Ankle boot'
# }