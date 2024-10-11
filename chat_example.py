from llm_connector import OpenAIChat
from utility import TextUtil

if __name__=="__main__":
    chat = OpenAIChat(OpenAIChat.OpenAIModel.GPT_4O_mini)
    while True:
        prompt = input()
        print(TextUtil.get_colored_text(chat.ask(prompt), TextUtil.TEXT_COLOR.Yellow))
