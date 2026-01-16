# Machine Learning Final Project Model Deployment
# Titanic Survival Guessing Game
# User will be met with a story from a titanic passenger
# They make a guess whether that person survived
# The guess is validated with an MLP Machine Learning model

import tkinter as tk

from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import torch
from models import MLPClassifier, MLP4DResidualClassifier

class MyApp(tk.Frame):
    def __init__(self, root):

        self.current_page_index = 0
        self.pages = [self.create_page_container, self.hint_page, self.game_page1, self.game_page2,self.game_page3,self.game_page4,self.game_page5,self.build_passenger_page,self.end_page]
        self.total_score = 0
        # These switch when user clicks yes or no buttons
        self.no_button = 0
        self.yes_button = 0
        self.current_features = []

        super().__init__(
            root
        )

        self.main_frame = self
        self.main_frame.pack(fill="both", expand=True)
        # Frame is 1 column, 0=column index, weight=1 means can stretch
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0,weight=1)

        self.load_main_widgets()

    def load_main_widgets(self):
        self.create_page_container()
        self.pages[self.current_page_index]()

    # When switching pages, clear page first
    def clear_frame(self,frame):
        for child in frame.winfo_children():
            child.destroy()

    # Model deployment!
    # Evaluates the features of the titanic passengers story
    # returning 1 or 0 for whether they survived or not
    def ml_evaluate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cpu":
            model: MLP4DResidualClassifier = torch.load("models/mlp_fourlayer_model.mdl", weights_only = False)
        else:
            model: MLP4DResidualClassifier = torch.load("models/mlp_fourlayer_model.mdl", weights_only = False, map_location=device)

        model.eval()
        x = torch.tensor([self.current_features], dtype=torch.float32)
        with torch.no_grad():
            x = x.to("cpu")
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            # Gives value within preds tensor
            # print(preds.item())
            return preds.item()

    def change_page(self):
        self.clear_frame(self.page_container)
        self.current_page_index += 1
        self.pages[self.current_page_index]()


    # When user clicks yes or no button, check with correct
    # answer from model
    def score(self):
        correct = self.ml_evaluate()
        if self.no_button == 1 and correct == 0:
            self.total_score += 1
            self.no_button = 0
            showinfo("Question Response", "Correct")
        elif self.no_button == 1 and correct == 1:
            self.no_button = 0
            showinfo("Question Response", "Incorrect")
        elif self.yes_button == 1 and correct == 1:
            self.total_score += 1
            self.yes_button = 0
            showinfo("Question Response", "Correct")
        elif self.yes_button == 1 and correct == 0:
            self.yes_button = 0
            showinfo("Question Response", "Incorrect")

    def no_btn_clicked(self):
        self.no_button = 1
        self.score()
        self.change_page()

    def yes_btn_clicked(self):
        self.yes_button = 1
        self.score()
        self.change_page()

    # Container for main menu page
    def create_page_container(self):
        self.page_container = tk.Frame(
            self.main_frame
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index = 1
            self.pages[self.current_page_index]()

        # Background image
        self.bg_image = ImageTk.PhotoImage(Image.open("imgs/titanic.jpg"))
        image_label = tk.Label(self.page_container, image=self.bg_image)
        image_label.place(x=0, y=0, relwidth=1, relheight=1)
        image_label.lower()

        nrows = 5

        # For rules label
        rules = """
The sinking of the Titanic in 1912 was a tragic event that took hundreds of lives.

It turns out that many factors influenced whether someone would survive the crash.

You will be introduced to former passengers of the Titanic. You will listen to their stories, investigate the scene, and determine whether you think they would survive the crash.


This is a Machine Learning game. Your guess will be run through a trained ML Model.
The model will decide whether you are correct, based on its training data.
        """

        self.page_container.columnconfigure(0,weight=0)
        # 2nd column fills rest of screen
        self.page_container.columnconfigure(1,weight=1)

        for i in range(nrows):
            # Add padding to start button
            if i == 4:
                self.page_container.rowconfigure(i,weight=1)
            else:
                self.page_container.rowconfigure(i,weight=0)

        # Makes container take up the whole screen
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # This stuff would be moved to a different function, this not structure but what
        # goes in the structure, and root should be changed
        title_txt = tk.Label(self.page_container, text="Titanic Guesser", font=("Helvetica", 30), fg="white",bg="#217fdd")
        title_txt.grid(row=0, column=1)
        author_txt = tk.Label(self.page_container, text="Olly Love, Nathan Singer, David Kelly", font=("Helvetica", 15), fg="white",bg="#217fdd")
        author_txt.grid(row=1,column=1)
        rule_btn = tk.Button(self.page_container, text="Rules", font=("Helvetica", 15), fg="white", bg="#217fdd")
        rule_btn.grid(row=2,column=0,sticky="w")
        # FORMAT ERROR - Why not going on the left?
        rule_txt = tk.Label(self.page_container, text=rules, font=("Helvetica", 12),fg="white",bg="#217fdd",wraplength=400,anchor="e",justify="left")
        rule_txt.grid(row=3,column=0,sticky="e")
        # Change scenes
        start_btn = tk.Button(self.page_container, text="Start", font=("Helvetica", 20), fg="white", bg="#217fdd",command=change_page)
        start_btn.grid(row=4,column=1)

    # Hint page
    def hint_page(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#000000"
        )

        def change_page():
            self.clear_frame(self.page_container)
            self.current_page_index = 2
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        hint = """
        To aid you in making your choices - women and children were prioritized
        access to lifeboats. The nicer ($$$) living quarters were closer to the top of the ship.
        Lifeboats were deployed from the top of the ship.
        """

        hint_txt = tk.Label(self.page_container,text=hint, font=("Helvetica", 20),wraplength=500,fg="white",bg="#000000",justify="center")
        hint_txt.grid(row=0, column=0,sticky="s",columnspan=2)
        continue_btn = tk.Button(self.page_container, text="Continue", font=("Helvetica", 20), fg="white", bg="#000000",command=change_page)
        continue_btn.grid(row=1,column=1, sticky="s")

    # First page of game, displaying passengers portrait,
    # story, guessing buttons, and button to advance game to the next page
    def game_page1(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Didn't Survive - PassengerId = 1, info from dataset + made up details on top
        # related to the dataset info
        # Copy and pasted feature vector for this passenger
        self.current_features = [3,0,22.000000,1,0,7.2500,0,0,0,0,0,0,0,0,1,0,0,1]
        story = """
        Story 1:
        Hi, I'm Mr. Owen Harris Braund.

        I heard about the Titanic all over the news and just had to go.

        I worked extra hours as a server to make just enough for a 3rd class ticket. Its hard finding work at my age as I'm only 22, not many people want to hire someone like me, especially with no university education.

        I worked very hard to be here, I even managed to grab an extra ticket for my brother Lewis.

        I see all these families around, I'm so grateful I don't have any kids to look after, that seems like a tough job. Though, some of those families are living on the upper decks, I'm in a shared cabin at the bottom of the ship.

        Theres always a compromise, but I'm enjoying my time here anyways.
        """

        # Titanic man 1
        self.portrait = ImageTk.PhotoImage(Image.open("imgs/titanicman1.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4)
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")

        # Here needs to connect to the model - must create functions storing user input
        # yes = 1, no = 0, if yes_btn.click() or something like that
        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A",command=self.no_btn_clicked)
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a",command=self.yes_btn_clicked)
        yes_btn.grid(row=2,column=2,sticky="s")

        score_txt = tk.Label(self.page_container,text="Score: " + str(self.total_score), font=("Helvetica", 12),fg="white",bg="#217fdd")
        score_txt.grid(row=3, column=1,sticky="w")

    def game_page2(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )


        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Survived - PassengerId = 3
        self.current_features = [3,1,26.000000,0,0,7.9250,0,0,0,0,0,0,0,0,1,0,0,1]
        story = """
        Story 2:
        How do you do? My names Miss. Laina Heikkinen.

        As you can tell, I'm not married. In fact I'm here all alone. My grandma gifted me a ticket here for my birthday.

        We aren't wealthy so I'm staying in a shared cabin on the lower decks, but it doesn't matter, its such a beautiful ship and I'm having a great time.

        I can't wait to tell my family all about the experience!
        """

        # Titanic woman 2
        self.portrait = ImageTk.PhotoImage(Image.open("imgs/titanicwoman2.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        # Change to grid
        # image_label.place(x=0, y=0, relwidth=1, relheight=1)
        # image_label.lower()
        image_label.grid(row=0,column=0,rowspan=4)
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")
        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A",command=self.no_btn_clicked)
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a",command=self.yes_btn_clicked)
        yes_btn.grid(row=2,column=2,sticky="s")
        score_txt = tk.Label(self.page_container,text="Score: " + str(self.total_score), font=("Helvetica", 12),fg="white",bg="#217fdd")
        score_txt.grid(row=3, column=1,sticky="w")

    def game_page3(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Survived - PassengerId = 18
        self.current_features = [2,0,30,0,0,13,0,0,0,0,0,0,0,0,1,0,0,1]
        story = """
        Story 3:
        Hello, I'm Mr. Charles Eugene Williams, but you can call me Charles.

        I'm taking a vacation from my ungrateful family, I work so hard, buy them tons of nice things, but all they do is complain.

        I heard about the Titanic on the news and quickly bought this 2nd class ticket, all the first class ones were sold out, or I woulda bought 2, one for me, and one for my bags.

        I'm close enough to all the amenities on the top floor, so I don't mind being where I am.
        """

        # Titanic man 3
        # Width  * height
        self.portrait = ImageTk.PhotoImage(Image.open("imgs/titanicman3.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4,sticky="w")
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")

        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A",command=self.no_btn_clicked)
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a",command=self.yes_btn_clicked)
        yes_btn.grid(row=2,column=2,sticky="s")

        score_txt = tk.Label(self.page_container,text="Score: " + str(self.total_score), font=("Helvetica", 12),fg="white",bg="#217fdd")
        score_txt.grid(row=3, column=1,sticky="w")


    def game_page4(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Survived - PassengerId = 44
        self.current_features = [2,1,3,1,2,41.5792,0,0,0,0,0,0,0,0,1,1,0,0]
        story = """
        Story 4:
        Goo goo ga ga.

        I'm Miss. Simone Marie Anne Andree Laroche!

        I'm only 3!

        My parents are amazing and buy me tons of nice things, like this 2nd class ticket to the Titanic!

        This ship is amazing!
        """

        self.portrait = ImageTk.PhotoImage(Image.open("imgs/titanicbaby.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4,sticky="w")
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")

        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A",command=self.no_btn_clicked)
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a",command=self.yes_btn_clicked)
        yes_btn.grid(row=2,column=2,sticky="s")

        score_txt = tk.Label(self.page_container,text="Score: " + str(self.total_score), font=("Helvetica", 12),fg="white",bg="#217fdd")
        score_txt.grid(row=3, column=1,sticky="w")


    def game_page5(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)
        self.page_container.columnconfigure(2,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        # Survived - PassengerId = 86
        self.current_features = [3,1,33,3,0,15.85,0,0,0,0,0,0,0,0,1,0,0,1]
        story = """
        Story 5:
        Well hey there, my names Mrs. Karl Alfred.

        My husband thought it would be nice to surprise the family with Titanic tickets, so here we are.

        Our quarters are cramped, but its a fabulous ship with tons to do.
        If only the kid didn't keep running away, but, at least were getting good exercise.
        """

        self.portrait = ImageTk.PhotoImage(Image.open("imgs/titanicmom.jpg").resize((450, 600), Image.Resampling.LANCZOS))
        image_label = tk.Label(self.page_container, image=self.portrait)
        image_label.grid(row=0,column=0,rowspan=4,sticky="w")
        story_txt = tk.Label(self.page_container,text=story, font=("Helvetica", 12),wraplength=500,fg="white",bg="#217fdd",anchor="w")
        story_txt.grid(row=0, column=1,columnspan=2,sticky="w")
        question_txt = tk.Label(self.page_container,text="Did they survive the Titanic?", font=("Helvetica", 12),fg="white",bg="#217fdd")
        question_txt.grid(row=1, column=1,columnspan=2,sticky="n")

        no_btn = tk.Button(self.page_container, text="No", font=("Helvetica", 20), fg="white", bg="#7A0A0A",command=self.no_btn_clicked)
        no_btn.grid(row=2,column=1,sticky="s")
        yes_btn = tk.Button(self.page_container, text="Yes", font=("Helvetica", 20), fg="white", bg="#0b711a",command=self.yes_btn_clicked)
        yes_btn.grid(row=2,column=2,sticky="s")

        score_txt = tk.Label(self.page_container,text="Score: " + str(self.total_score), font=("Helvetica", 12),fg="white",bg="#217fdd")
        score_txt.grid(row=3, column=1,sticky="w")


    # Final page showing users final score
    def end_page(self):
        self.page_container = tk.Frame(
            self.main_frame,
        )

        self.bg_image = ImageTk.PhotoImage(Image.open("imgs/titanic.jpg"))
        image_label = tk.Label(self.page_container, image=self.bg_image)
        image_label.place(x=0, y=0, relwidth=1, relheight=1)
        image_label.lower()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        question_txt = tk.Label(self.page_container,text="The End!", font=("Helvetica", 30),fg="white",bg="#217fdd")
        question_txt.grid(row=0, column=0)

        score_txt = tk.Label(self.page_container,text="Final Score: " + str(self.total_score), font=("Helvetica", 30),fg="white",bg="#217fdd")
        score_txt.grid(row=1, column=0)

        if self.custom_survival:
            bonus_label = "The custom passenger survived!"
        else:
            bonus_label = "The custom passenger did not survive..."

        bonus_txt = tk.Label(self.page_container,text=bonus_label, font=("Helvetica", 15),fg="white",bg="#217fdd")
        bonus_txt.grid(row=2, column=0)

    # Bonus build passenger page
    def build_passenger_page(self):
        self.page_container = tk.Frame(
            self.main_frame,
            bg="#217fdd"
        )

        # Evaluate character survival and change page
        def submit_input():
            self.current_features = [
                int(entry.get()),
                int(entry2.get()),
                int(entry3.get()),
                int(entry4.get()),
                int(entry5.get()),
                int(entry6.get()),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1
            ]
            survived = self.ml_evaluate()
            if survived == 1:
                self.custom_survival = 1
            else:
                self.custom_survival = 0

            self.clear_frame(self.page_container)
            self.current_page_index += 1
            self.pages[self.current_page_index]()

        self.page_container.columnconfigure(0,weight=1)
        self.page_container.columnconfigure(1,weight=1)

        self.page_container.rowconfigure(0,weight=1)
        self.page_container.rowconfigure(1,weight=1)
        self.page_container.rowconfigure(2,weight=1)
        self.page_container.rowconfigure(3,weight=1)
        self.page_container.rowconfigure(4,weight=1)
        self.page_container.rowconfigure(5,weight=1)
        self.page_container.rowconfigure(6,weight=1)
        self.page_container.rowconfigure(7,weight=1)

        self.page_container.grid(column=0, row=0, sticky=tk.NSEW)

        title_txt = tk.Label(self.page_container,text="Bonus: Create a Passenger", font=("Helvetica", 30),fg="white",bg="#217fdd")
        title_txt.grid(row=0, column=0,columnspan=2)
        entry1_txt = tk.Label(self.page_container,text="Class (1/2/3): ", font=("Helvetica", 12),fg="white",bg="#217fdd")
        entry1_txt.grid(row=1, column=0)
        entry = tk.Entry(self.page_container, width=30)
        entry.grid(row=1,column=1)

        entry2_txt = tk.Label(self.page_container,text="Sex (0 for male, 1 for female): ", font=("Helvetica", 12),fg="white",bg="#217fdd")
        entry2_txt.grid(row=2, column=0)
        entry2 = tk.Entry(self.page_container, width=30)
        entry2.grid(row=2,column=1)

        entry3_txt = tk.Label(self.page_container,text="Age: ", font=("Helvetica", 12),fg="white",bg="#217fdd")
        entry3_txt.grid(row=3, column=0)
        entry3 = tk.Entry(self.page_container, width=30)
        entry3.grid(row=3,column=1)

        entry4_txt = tk.Label(self.page_container,text="# of Siblings/Spouse on board: ", font=("Helvetica", 12),fg="white",bg="#217fdd")
        entry4_txt.grid(row=4, column=0)
        entry4 = tk.Entry(self.page_container, width=30)
        entry4.grid(row=4,column=1)

        entry5_txt = tk.Label(self.page_container,text="# of Parents/Children on board: ", font=("Helvetica", 12),fg="white",bg="#217fdd")
        entry5_txt.grid(row=5, column=0)
        entry5 = tk.Entry(self.page_container, width=30)
        entry5.grid(row=5,column=1)

        entry6_txt = tk.Label(self.page_container,text="Fare Paid (digits only) $: ", font=("Helvetica", 12),fg="white",bg="#217fdd")
        entry6_txt.grid(row=6, column=0)
        entry6 = tk.Entry(self.page_container, width=30)
        entry6.grid(row=6,column=1)

        submit_btn = tk.Button(self.page_container, text="Next", font=("Helvetica", 20), fg="white", bg="#217fdd",command=submit_input)
        submit_btn.grid(row=7,column=0)

root = tk.Tk()
root.title('Titanic Guesser')
root.geometry("1000x650")
root.resizable(width=False,height=False)
app_instance = MyApp(root)
root.mainloop()
