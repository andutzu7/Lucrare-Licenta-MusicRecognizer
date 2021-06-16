from ModelClass.Model import Model
from utils.pipeline import *

from tkinter import *
import time
import threading
import imageio
from PIL import Image, ImageTk

model = Model.load('./Models/working_audio_classification.h5')
root = Tk()

def predict(link):
    """
    The method downloads the song at the link and predicts its label.

    Args: link(string): The link to the video.

    """
    # Downloading the song
    download_song_from_youtube(link)
    # Computing the file path
    file_path = os.path.join('./tempdownload', os.listdir('./tempdownload')[0])
    # Predict the label
    result = get_song_label(file_path, model)
    # Get the most predicted label
    prediction = max(result.items(), key=operator.itemgetter(1))[0]
    print(f'''Out of {sum(result.values())} guesses I predicted your song {result["Piano"]} times as a piano song and {result["Other"]} times as other instruments song.\n
    My prediction is {prediction}
    ''')
    # Return the prediction
    return prediction

def process_song_from_input(textBox,my_label):
    """
    Exstract the link from the textbox

    Args: textBox(Entry): TextBox containing the link to the song
          my_label(Label): Label for the text output

    """
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
    """
    The ResizingCanvas class makes the GUI responsive on App Resize event.

    Sources: https://stackoverflow.com/questions/22835289/how-to-get-tkinter-canvas-to-dynamically-resize-to-window-width
    """
    def __init__(self,parent,**kwargs):
        """
        Sources: https://stackoverflow.com/questions/22835289/how-to-get-tkinter-canvas-to-dynamically-resize-to-window-width
        """
        Canvas.__init__(self,parent,**kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self,event):
        """
        Determine the ratio of old width/height to new width/height
        Args: *event(event): The resize event
        Sources:  *https://stackoverflow.com/questions/22835289/how-to-get-tkinter-canvas-to-dynamically-resize-to-window-width
        """
        wscale = float(event.width)/self.width
        hscale = float(event.height)/self.height
        self.width = event.width
        self.height = event.height
        # Resize the canvas 
        self.config(width=self.width, height=self.height)
        # Rescale all the objects tagged with the "all" tag
        self.scale("all",0,0,wscale,hscale)
  

def callback(event):
    """
    Callback for making the text selectable with ctrl-a

    Source:     *https://www.python-course.eu/tkinter_events_binds.php     
                *https://stackoverflow.com/questions/41477428/ctrl-a-select-all-in-entry-widget-tkinter-python 

    """
    root.after(50, select_all, event.widget)

def select_all(widget):
    """
    Widget for making the text selectable with ctrl-a
    Source:  *https://www.python-course.eu/tkinter_events_binds.php
             *https://stackoverflow.com/questions/41477428/ctrl-a-select-all-in-entry-widget-tkinter-python

    """
    widget.select_range(0, 'end')
    widget.icursor('end')


def generate_video(label):
    """
    Feed the proper song to the GUI
    Args: label(string): Predicted label
    """
    if label == 'Piano':
        return imageio.get_reader('piano.wmv')
    else:
        return imageio.get_reader('nicovala.wmv')
        

def stream(label,video):

    """
    Function that displays the video.
    Args: *label: Video label
          *video: Video file
    Sources:    *https://stackoverflow.com/questions/36635567/tkinter-inserting-video-into-window
                *https://stackoverflow.com/questions/16996432/how-do-i-bind-the-enter-key-to-a-function-in-tkinter
    """
    for image in video.iter_data():
        frame_image = ImageTk.PhotoImage(Image.fromarray(image))
        label.config(image=frame_image)
        label.image = frame_image
        # Slow down the video rate
        time.sleep(0.05)

if __name__ == "__main__":

    # Initialize the frame
    myframe = Frame(root)
    # Add the application title
    root.title(" Music Recognizer ")
    # Make the GUI responsize
    myframe.pack(fill=BOTH, expand=YES)
    mycanvas = ResizingCanvas(myframe,width=850, height=400,  highlightthickness=0)
    mycanvas.pack(fill=BOTH, expand=YES)
    
    # Text label of the message for the user
    label = Label(root, text='Insert the Youtube link of the song you want to classify.')
    label.config(font=("Courier",16))
    label.place(relx=0.4, rely=0.05, anchor='s')

    # Initialize the textInput box for the link
    txtVar = StringVar(None)
    userIn = Entry(root, textvariable = txtVar, width = 100,font=("Courier",16))
    # Make the text selectable with ctrl-a
    userIn.bind('<Control-a>', callback)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    userIn.place(width =screen_width/3,height = screen_height/35,relx=0.65, rely=0.07, anchor='n')

    # Label for the output
    my_label = Label(root)
    my_label.place(relx = 0.5, rely = 0.6, anchor = CENTER)


    # Button for entering the link
    submitButton=Button(root, height=1, width=15, text="Submit", 
                        command=lambda: process_song_from_input(userIn,my_label))
    # Binding the Enter key for submiting the input
    userIn.bind("<Return>", lambda x : process_song_from_input(userIn,my_label))
    submitButton.place(relx = 0.65, rely = 0.13, anchor = CENTER)


    l = Label(root, text = "Music Classificator powered by Neural Networks")
    l.config(font =("Courier", 21))

    l.pack()
    # Run the application
    root.mainloop()