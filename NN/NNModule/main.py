from ModelClass.Model import Model
from utils.pipeline import *

if __name__ == "__main__":

    model = Model.load('./Models/working_audio_classification.h5')
    download_song_from_youtube('https://www.youtube.com/watch?v=HhqET0xuU0I')
    file_path = os.path.join('./tempdownload', os.listdir('./tempdownload')[0])
    result = get_song_label(file_path, model)
    print(f'''Out of {sum(result.values())} guesses I predicted your song {result["Piano"]} times as a piano song and {result["Other"]} times as other instruments song.\n
    My prediction is {max(result.items(), key=operator.itemgetter(1))[0]}
    ''')