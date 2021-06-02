# o sa am nevoie si eu de rahatu ala cu training(argumentu nefolosit)
# pentru maaxpooling si conv2d metoda e ca primesc batchu la forma (N , C , W, H) si atributu input shape ia shape[1:], dar trebuie sa
# greseala e la input batch handling si faza cu channels first
# si faza cu training loop, dupa ce se face forward si backprop trebuie reshapeuit arrayul(in model training loop)
# am in vedere ca inputul primit dinainte in retea sa fie de forma asta sau sa i se faca reshape .
from Layers.InputLayer import InputLayer
from ModelClass.Model import Model
from Metrics.CategoricalAccuracy import CategoricalAccuracy
from Metrics.CategoricalCrossentropy import CategoricalCrossentropy
from Optimizers.OptimizerAdam import OptimizerAdam
from Activations.ActivationReLu import ActivationReLu
from Activations.ActivationSoftmax import ActivationSoftmax
from Layers.DropoutLayer import DropoutLayer
from Layers.DenseLayer import DenseLayer
from Layers.BatchNormalizationLayer import BatchNormalization
from Layers.FlattenLayer import Flatten
from Layers.MaxPooling2DLayer import MaxPooling2D
from Layers.Conv2DLayer import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ModelClass.DataGenerator import DataGenerator
from glob import glob                
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.io import wavfile

def create_dataset(path):

    wav_paths = glob('{}/**'.format(path), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = os.listdir(path)
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.1,
                                                                  random_state=0)
    return wav_train, wav_val, label_train, label_val


if __name__ == "__main__":

    # path = './clean/piano'
    # create_dataset(path)
    # sr = 16000
    # dt = 1.0
    # nr_classes = 2
    # batch_size = 32
    # wav_train, wav_val, label_train, label_val = create_dataset(path)

    # # Creating the data generators
    # tg = DataGenerator(wav_train, label_train, sr, dt,
    #                    nr_classes, batch_size=batch_size)
    # vg = DataGenerator(wav_val, label_val, sr, dt,
    #                    nr_classes, batch_size=batch_size)

    # # il = InputLayer()
    # # il.forward(tg.__getitem__(0)[0])
    # # print(il.output.shape)
    # model = Model()
    # # Add layers
    # model.add(BatchNormalization())
    # model.add(Conv2D(8, (7, 7)))
    # model.add(ActivationReLu())
    # model.add(MaxPooling2D(pool_shape=(2, 2), padding='same'))
    # model.add(ActivationReLu())
    # model.add(Conv2D(16, (5, 5)))
    # model.add(ActivationReLu())
    # model.add(MaxPooling2D(pool_shape=(2, 2), padding='same'))
    # model.add(ActivationReLu())
    # model.add(Conv2D(16, (3, 3)))
    # model.add(ActivationReLu())
    # model.add(MaxPooling2D(pool_shape=(2, 2), padding='same'))
    # model.add(ActivationReLu())
    # model.add(Conv2D(32, (3, 3)))
    # model.add(ActivationReLu())
    # model.add(MaxPooling2D(pool_shape=(2, 2), padding='same'))
    # model.add(ActivationReLu())
    # model.add(Flatten())
    # model.add(DropoutLayer(0.2))
    # model.add(DenseLayer(131072, 64))
    # model.add(ActivationReLu())
    # model.add(DenseLayer(64, 2))
    # model.add(ActivationSoftmax())

    # # Set loss, optimizer and accuracy objects
    # model.set(
    #     loss=CategoricalCrossentropy(),
    #     optimizer=OptimizerAdam(decay=5e-5),
    #     accuracy=CategoricalAccuracy()
    # )
    # # Finalize the model
    # model.finalize()
    # # Train the model
    # model.train(train_generator=tg, validation_generator=vg,
    #             epochs=5
    #             , batch_size=batch_size, print_every=1)
    # model.save('working_audio_classification2.h5')
    dataset_labels={0:'Other',
                    1:'Piano'}
    audio_file = wavfile.read('./hexagon16_0.wav')

    X = np.empty((1, int(16000*1.0), 1),dtype=np.float32)

    Y = np.empty((1, 2), dtype=np.float32)
    X[0, ] = audio_file[1].reshape(-1, 1)
    
    model = Model.load('working_audio_classification.h5')
    confidences = model.predict(X)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)
    print(confidences)
    # Get label name from label index
    prediction = dataset_labels[predictions[0]]
    print(prediction)

