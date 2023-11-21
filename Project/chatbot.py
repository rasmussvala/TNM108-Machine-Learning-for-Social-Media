import pandas as pd
import random

# Read the CSV file into a pandas DataFrame
data = pd.read_csv("Conversation.csv")

# Convert the DataFrame into a dictionary
conversations = data.set_index("question").T.to_dict("list")


# Function to generate bot responses
def chatbot(question):
    return random.choice(
        conversations.get(question, ["Sorry, I didn't understand that."])
    )


# Example conversation loop
print("Bot: Hello! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = chatbot(user_input)
    print("Bot:", response)
