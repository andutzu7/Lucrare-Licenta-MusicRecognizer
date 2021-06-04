from ModelClass.Model import Model
from utils.pipeline import *

if __name__ == "__main__":

    model = Model.load('./Models/working_audio_classification.h5')
    result = get_song_label('Test data/Initial/shosta.wav',model)
    print(f'''Out of {sum(result.values())} guesses I predicted your song {result["Piano"]} times as a piano song and {result["Other"]} 
    times as other instruments song.\n
    My prediction is {max(result.items(), key=operator.itemgetter(1))[0]}
    ''')