import aiml
import os
import sqlite3 

# Create the kernel and learn AIML files
kernel = aiml.Kernel()

#initialize database values so connection can be made with databse
sqlite_file = '../database/chatbotDB.sqlite'# name of the sqlite database file
table_name = 'chatbot_convo_table'  # name of the table to be created

column1 = 'ID' #primary key
column2 = 'conversation_ID'  #id of each conversation
column3 = 'chatbot_type' #chat type will determine if chat was from 1 = AIML chatbot, or 2 = RNN chatbot
column4 = 'message_type' #message type column will determine if message is from user or chatbot 1= user, or 2 = chatbot
column5 = 'message' #message content

#function to insert chat messages to database, with corresponding metadata value
def insertInputMessage(conversationID, messagetype,inputmessage):
    # Connecting to the database file
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    try:
        params = (conversationID, 1, messagetype, inputmessage)
        c.execute("INSERT INTO chatbot_convo_table (conversation_ID, chatbot_type, message_type, message) VALUES (?, ?, ?, ?)",params)
    except sqlite3.IntegrityError:
        print('ERROR: ID already exists in PRIMARY KEY column {}'.format(id_column))

    # Committing changes and closing the connection to the database file
    conn.commit()
    conn.close()

#function to determine which conversation ID to assign to conversation that is occuring. it looks for the last conversation id, and adds 1
def assignConversationID():
    # Connecting to the database file
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    #get the last conversation_id value
    c.execute("SELECT MAX(conversation_ID) FROM chatbot_convo_table;")
    value = c.fetchone()[0]

    if value is None:
        return 1
    else:
        return value + 1

    conn.close()


v = assignConversationID()

kernel.bootstrap(learnFiles = "std-startup.xml", commands = "load aiml b")

# Press CTRL-C to break this loop
while True:
    message = input("Enter your message to the bot: ")
    if message == "quit" or message == "QUIT" or message == "Quit":
        exit()
    else:
        bot_response = kernel.respond(message)
        insertInputMessage(v, 1, message)
        print(bot_response)
        insertInputMessage(v, 2, bot_response)
        