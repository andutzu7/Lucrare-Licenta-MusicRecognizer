# add no playlist
from ModelClass.Model import Model
from utils.pipeline import *

from tkinter import *
import threading
import imageio
from PIL import Image, ImageTk

model = Model.load('./Models/working_audio_classification.h5')
root = Tk()

def predict(link):
    download_song_from_youtube(link)
    file_path = os.path.join('./tempdownload', os.listdir('./tempdownload')[0])
    result = get_song_label(file_path, model)
    prediction = max(result.items(), key=operator.itemgetter(1))[0]
    print(f'''Out of {sum(result.values())} guesses I predicted your song {result["Piano"]} times as a piano song and {result["Other"]} times as other instruments song.\n
    My prediction is {prediction}
    ''')
    return prediction

def process_song_from_input(textBox,my_label):
    inputValue=textBox.get()
    textBox.delete(0, "end")
    predicted_label = predict(str(inputValue))
    label = Label(root, text=f'The given song falls into the {predicted_label} category.')
    label.config(font=("Courier",16))
    label.place(relx=0.4, rely=0.3, anchor='s')

    video = generate_video(predicted_label)
    thread = threading.Thread(target=stream, args=(my_label,video))
    thread.daemon = 1
    thread.start()

# a subclass of Canvas for dealing with resizing of windows
class ResizingCanvas(Canvas):
    def __init__(self,parent,**kwargs):
        Canvas.__init__(self,parent,**kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self,event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width)/self.width
        hscale = float(event.height)/self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas 
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all",0,0,wscale,hscale)
  

def callback(event):
    root.after(50, select_all, event.widget)

def select_all(widget):
    widget.select_range(0, 'end')
    widget.icursor('end')


def generate_video(label):
    if label == 'Piano':
        return imageio.get_reader('piano.wmv')
    else:
        return imageio.get_reader('nicovala.wmv')
        

def stream(label,video):

    for image in video.iter_data():
        frame_image = ImageTk.PhotoImage(Image.fromarray(image))
        label.config(image=frame_image)
        label.image = frame_image

if __name__ == "__main__":

    myframe = Frame(root)
    root.title(" Music Recognizer ")
    myframe.pack(fill=BOTH, expand=YES)
    mycanvas = ResizingCanvas(myframe,width=850, height=400,  highlightthickness=0)
    mycanvas.pack(fill=BOTH, expand=YES)
    
    label = Label(root, text='Insert the Youtube link of the song you want to classify.')
    label.config(font=("Courier",16))
    label.place(relx=0.4, rely=0.05, anchor='s')

    txtVar = StringVar(None)

    userIn = Entry(root, textvariable = txtVar, width = 100,font=("Courier",16))
    userIn.bind('<Control-a>', callback)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    userIn.place(width =screen_width/3,height = screen_height/35,relx=0.65, rely=0.07, anchor='n')

    my_label = Label(root)
    my_label.place(relx = 0.5, rely = 0.6, anchor = CENTER)


    submitButton=Button(root, height=1, width=15, text="Submit", 
                        command=lambda: process_song_from_input(userIn,my_label))
    userIn.bind("<Return>", lambda x : process_song_from_input(userIn,my_label))
    submitButton.place(relx = 0.65, rely = 0.13, anchor = CENTER)

    l = Label(root, text = "Music Classificator powered by Neural Networks")
    l.config(font =("Courier", 21))

    l.pack()
    root.mainloop()