from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen

# Wrapper to print boxen objects to terminal
def boxen_print(*args, **kwargs):
    print(boxen(*args,**kwargs))


# Reference
# class BaseCallbackHandler(
#     LLMManagerMixin,
#     ChainManagerMixin,
#     ToolManagerMixin,
#     RetrieverManagerMixin,
#     CallbackManagerMixin,
#     RunManagerMixin,

# Reference - message.types
# human, ai, function, system

class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n ======== Sending Messages ========\n\n")
        for message in messages[0]:
            if message.type == "system":
                boxen_print(message.content, title=message.type,color="yellow")
            elif message.type == "human":
                boxen_print(message.content, title=message.type,color="green")
            # An ai messages where AI is running a function    
            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                boxen_print(
                    f"Running tool {call['name']} with args {call['arguments']}",
                    title=message.type,
                    color="cyan"
                )
            # a standard ai assistant message 
            elif message.type == "ai":
                boxen_print(message.content, title=message.type,color="blue")
            # 
            elif message.type == "function":
                boxen_print(message.content, title=message.type, color="purple")
            else:
                boxen_print(message.content, title=message.type)
                