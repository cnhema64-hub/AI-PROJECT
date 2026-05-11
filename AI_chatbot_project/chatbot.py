print("Welcome to AI Chatbot")
while True:
   user = input("you: ").lower()
   if user=="hello" or user =="hi":
       print("Bot: Hi")
   elif user=="how are you" or  user =="how are u" or user =="how r u":
       print("Bot: I am doing great How can i help you?")
   elif user =="can you help me":
       print("Bot:yes Tell me your problem or dought")
   elif user ==" what is ai" or user =="tell me about ai":
       print("""Bot:Artificial intelligence (AI) is a technology that allowes computers and mchines to think,learn, and make decisions like humans. It is used in chatbots, self-driving cars, recommendations system, and virtual assistents like Siri and Alexa.""")
   elif user =="what is python" or user =="python meaning":
       print("""Bot: python is a popular programing language used for wed development, AI, data science, and automation. It is easy to learn and very powerful.""")
   elif user =="what is sql" or user =="sql":
       print("""Bot: Sql stands for Strature query language .which is pronounced as a SEQUEL.This language is used to communicate with oracle data-base. It is a commend based language. It is a case insensitive language.Every commend must start with verd and end with semicolon.EX: SELECT,UPDATE,DELETE,DROP,INSERT & etc.""" )
   elif user=="what are the sub languages in sql":
       print("""Bot:  sub languages are DDL (Data Defination Language), DML (Data  Manipulation Language), DCL (Data Control Language), TCL  (Transaction Control Language), DQL/DRL  (Data Retrive/Query Language). """)
   elif user =="bye" or user =="ok bye":
       print("Bot: bye see you tomorrow and keep learning")
       break
   else: 
       print("Bot: Sorry I did't Understand")