from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

print("_____Welcome type 0 to exit from my application_____")
print("🤖 AI Chatbot")
print("1: funny")
print("2: Angry")
print("3: Sad")
print("4: Romantic")

personalities={
    "1":"You are a funny chatbot who responds with humor",
    "2":"You are a angry chatbot who responds in an irritated tone",
    "3":"You are a sad chatbot who responds in a melancholic tone",
    "4":"You are a romantic chatbot who responds in a romantic manner",

}
choice =input("\n Choose personality(1-4):")
if choice not in personalities:
    print("invalid choice!")
    exit()

messages =[
 SystemMessage(content=personalities[choice])
]
while True:
    prompt = input("YOU:")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        print("thank you for using my application")
        break
   
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content[0]["text"]))
    print("Bot:",response.content[0]["text"])
    # print(messages)
