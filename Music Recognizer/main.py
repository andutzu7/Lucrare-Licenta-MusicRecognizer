# Add exception handling for invalid lincss
# add no playlist
from ModelClass.Model import Model
from utils.pipeline import *

from tkinter import *

model = Model.load('./Models/working_audio_classification.h5')
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
  
root = Tk()

def callback(event):

    root.after(50, select_all, event.widget)

def select_all(widget):
    widget.select_range(0, 'end')
    widget.icursor('end')

def predict(link):
    download_song_from_youtube(link)
    file_path = os.path.join('./tempdownload', os.listdir('./tempdownload')[0])
    print("################################")
    print(file_path)
    result = get_song_label(file_path, model)
    print(f'''Out of {sum(result.values())} guesses I predicted your song {result["Piano"]} times as a piano song and {result["Other"]} times as other instruments song.\n
    My prediction is {max(result.items(), key=operator.itemgetter(1))[0]}
    ''')

def process_song_from_input(textBox):
    inputValue=textBox.get()
    textBox.delete(0, "end")
    predict(str(inputValue))

def main():

    myframe = Frame(root)
    root.title(" Music Recognizer ")
    myframe.pack(fill=BOTH, expand=YES)
    mycanvas = ResizingCanvas(myframe,width=850, height=400,  highlightthickness=0)
    mycanvas.pack(fill=BOTH, expand=YES)
    
    # Create text widget and specify size.
  
    # Create label
    label = Label(root, text='Insert the Youtube link of the song you want to classify.')
    label.config(font=("Courier",16))
    label.place(relx=0.4, rely=0.05, anchor='s')

    txtVar = StringVar(None)

    userIn = Entry(root, textvariable = txtVar, width = 100,font=("Courier",16))
    userIn.bind('<Control-a>', callback)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    userIn.place(width =screen_width/3,height = screen_height/35,relx=0.65, rely=0.07, anchor='n')

    submitButton=Button(root, height=1, width=15, text="Submit", 
                        command=lambda: process_song_from_input(userIn))
    userIn.bind("<Return>", lambda x : process_song_from_input(userIn))
    submitButton.place(relx = 0.65, rely = 0.13, anchor = CENTER)


    l = Label(root, text = "Music Classificator powered by Neural Networks")
    l.config(font =("Courier", 21))

    l.pack()
    root.mainloop()

if __name__ == "__main__":

    main()