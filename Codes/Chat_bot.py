from nltk.chat.util import Chat, reflections
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you today ?",]
    ],
     [
        r"what is your name ?",
        ["My name is Chatty and I'm a chatbot ?",]
    ],
    [
        r"how are you ?",
        ["I'm doing good\nHow about You ?",]
    ],
    [
        r"(.*)issue(.*)",
        ["I can recommend services to resolve your issues.\nWhat is the issue?",]
    ],
    [
        r"(.*)storage(.*)",
        ["I would suggest you to try our 'Optimize for storage' service.It shall improve your experience",]
    ],
    [
        r"(.*)damaged(.*)",
        ["I would suggest you to try our 'Accidental Damage service' service.It shall protect your products",]
    ],
    [
        r"(.*)technical(.*)",
        ["I would suggest you to try our 'TechDirect' service.It shall improve your experience",]
    ],
    [
        r"(.*)travel(.*)",
        ["I would suggest you to try our 'Myservice360' service.It shall improve your experience",]
    ],
    [
        r"(.*)feedback(.*)",
        ["Your feedback has been noted. We at Dell, are committed to looking into your issues and enhancing your experience",]
    ],

    [
        r"(.*)not satisfied(.*)",
        ["I am sure our customer service can help you. You can contact them using - +91 8173456782",]
    ],

    [
        r"sorry (.*)",
        ["Its alright","Its OK, never mind",]
    ],
    [
        r"i'm (.*) doing good",
        ["Nice to hear that","Alright :)",]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there",]
    ],
    [
        r"what (.*) want ?",
        ["I want to enhance your experience",]
        
    ],
 [
        r"quit",
        ["Bye take care. See you soon :) ","It was nice talking to you. See you soon :)"]
],
]
def murphy():
    print("Hi, I'm your Murphy and I am here to help you :)\nLet me know how can I help you. Type quit to leave ") #default message at the start
    chat = Chat(pairs, reflections)
    chat.converse()
if __name__ == "__main__":
    murphy()