import openai
import chromadb
import os

SUMMARY_MESSAGE = "Summarize the following context: "
SYSTEM_CONTEXT_MESSAGE = "You are a helpful assistant who has been given the ability to recall context from previous conversations. Each message will include a user message and context. You should use the context to help you respond to the user message."
STANDARD_CONTEXT_MESSAGE = "Here is the context for the following user query: "


class ContextAnchoring:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./embeddings")
        self.collection = self.client.get_or_create_collection(name="context_embeddings")
        self.count = self.collection.count()
  
    def add_anchor(self, message, author):
        self.count += 1
        self.collection.add(documents=message, ids="id"+str(self.count), metadatas={"author": author})
  
    def check_anchor(self, message):
        return self.collection.query(query_texts=message)
    
    def summarize_context(self, context):
        messages = [{"role": "system", "content": SUMMARY_MESSAGE + context}]
        assistant_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return assistant_response.choices[0].message.content

class ChatBot:
    def __init__(self):
        self.context_handler = ContextAnchoring()
        self.conversation_history = [{"role": "system", "content": SYSTEM_CONTEXT_MESSAGE}]

    def receive_message(self, message):
        self.conversation_history.append({"role": "user", "content": message})
        query = self.context_handler.check_anchor(message)
        documents = query['documents']
        
        # Uncomment if you would like to see the documents and their distances to the user's message
        # print()
        # print(documents, "\n", query['distances'])
        # print()
        
        context = self.create_context(query, documents)

        if len(context) > 0:
            context = self.context_handler.summarize_context(context)
            self.conversation_history.insert(-1, {"role": "assistant", "content": STANDARD_CONTEXT_MESSAGE + context})
        
        return self.generate_response(message)

    def generate_response(self, message):
        assistant_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.conversation_history
        )
        
        self.conversation_history.append({"role": "assistant", "content": assistant_response.choices[0].message.content})
        
        self.context_handler.add_anchor(message, "user")

        return assistant_response

    def create_context(self, query, documents):
        context = ""
        for i in range(len(documents[0])):
            if query['distances'][0][i] < 1.3:
                # if the document already is present in the content of the previous messages, then don't add it to the context
                in_history = False
                for j in range(len(self.conversation_history)):
                    if documents[0][i] in self.conversation_history[j]['content']:
                        in_history = True
                        break
                if not in_history:
                    context += " --- " + str(documents[0][i])
        return context
      
      
openai.api_key = os.environ['OPENAI_API_KEY']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

chat_bot = ChatBot()

while (True):
    response = chat_bot.receive_message(input("You: "))
    print("Bot: " + response.choices[0].message.content)