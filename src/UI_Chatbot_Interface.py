from tkinter import *
from PredictResponse import provide_res_to_ui
from VoiceBasedLogic import listen

DBS_BOT_NAME = "DBS Research Chatbot"

FONT_FAMILY = "Helvetica 14" 
FONT_FAMILY_BOLD = "Poppins 13 bold"

FONT_COLOR_VAR = "#2C201D"
FOOTER_BACK_COLOR = "#F6F8F9"
BODY_BACK_COL = "#FCFCFC"


class DBSResearchChatbotApp:    
    def __init__(self):
        self.window = Tk()
        self._setup_main_chat_window()

    def run_chatbot_ui(self):
        self.window.mainloop()

    def _setup_main_chat_window(self):
        self.window.title("DBS Research Chatbot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=460, height=500, bg=BODY_BACK_COL)

        lbl_welcome = Label(self.window, bg=BODY_BACK_COL, fg = FONT_COLOR_VAR, text="Hi there! Welcome to DBS’s chatbot assistant", font=FONT_FAMILY_BOLD, pady=10)
        lbl_welcome.place(relwidth=1)

        lbl_line = Label(self.window, width=430, bg = FOOTER_BACK_COLOR)
        lbl_line.place(width=2, rely=0.08, relheight=0.014)

        self.txt_dbs_widget = Text(self.window, width=18, height=2, bg = BODY_BACK_COL, fg=FONT_COLOR_VAR, font=FONT_FAMILY, padx=5, pady=5)
        self.txt_dbs_widget.place(relheight=0.746, relwidth=1, rely=0.07)
        self.txt_dbs_widget.configure(cursor="arrow", state=DISABLED)

        scrollbar_dbs_chatbot =Scrollbar(self.txt_dbs_widget)
        scrollbar_dbs_chatbot.place(relheight=1, relx=0.974)
        scrollbar_dbs_chatbot.configure(command=self.txt_dbs_widget.yview)

        lbl_footer = Label(self.window, bg=FOOTER_BACK_COLOR, height=80)
        lbl_footer.place(relwidth=1,rely=0.825)

        self.txt_entry = Entry(lbl_footer, bg=BODY_BACK_COL, fg=FONT_COLOR_VAR, font=FONT_FAMILY)
        self.txt_entry.place(relwidth=0.62, relheight=0.05, rely=0.007, relx=0.010)
        self.txt_entry.focus()
        self.txt_entry.bind("<Return>",self._on_enter_pressed_event)

        btn_send = Button(lbl_footer, text="Send", font=FONT_FAMILY_BOLD, width=10, bg=FOOTER_BACK_COLOR, 
                             command=lambda:self._on_enter_pressed_event(None))
        btn_send.place(relx=0.63, rely=0.007, relheight=0.05, relwidth=0.16)

        mic_button = Button(lbl_footer, text="Speak",width=10, bg=FOOTER_BACK_COLOR, font=FONT_FAMILY_BOLD,
                            command=self.voice_input)        
        mic_button.place(relx=0.79, rely=0.007, relheight=0.05, relwidth=0.20)

    def _on_enter_pressed_event(self, event):
        msg = self.txt_entry.get()
        self._insert_message_event(msg,"You")

    def _insert_message_event(self, msg, sender):
        if not msg:
            return

        self.txt_entry.delete(0,END)
        msg1 = f"{sender}:{msg}\n\n"
        self.txt_dbs_widget.configure(state=NORMAL)
        self.txt_dbs_widget.insert(END,msg1)
        self.txt_dbs_widget.configure(state=DISABLED)

        msg2 = f"{DBS_BOT_NAME}:{provide_res_to_ui(msg)}\n\n"
        #msg2 = f"{DBS_BOT_NAME}\n"
        self.txt_dbs_widget.configure(state=NORMAL)
        self.txt_dbs_widget.insert(END,msg2)
        self.txt_dbs_widget.configure(state=DISABLED)        

        self.txt_dbs_widget.see(END)

    def voice_input(self):
        query_listen = listen()
        if query_listen:
            self._insert_message_event(query_listen, "You")
#app execution code. 
if __name__ == "__main__":
    app = DBSResearchChatbotApp()
    app.run_chatbot_ui()
