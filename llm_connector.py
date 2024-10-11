from __future__ import annotations

import openai
from enum import StrEnum
import copy

class LLMConnector:
    class Role(StrEnum):
        Assistant = "assistant"
        User = "user"
        System = "system"
    last_call_timestamp: dict[str, float] = {}
    tokens_used_this_minute: dict[str, int] = {}
    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier
    
    def _get_response(self, request) -> str:
        raise Exception("not implemented yet")
    
    def _prompt_to_request(self, prompt: str) -> object:
        raise Exception("not implemented yet")
    
    def inject(self, role: Role, text: str) -> None:
        raise Exception("not implemented yet")
    
    def copy(self) -> LLMConnector:
        raise Exception("not implemented yet")

    def ask(self, prompt: str) -> str:
        request = self._prompt_to_request(prompt)
        response = self._get_response(request)
        return response
    
class OpenAIChat(LLMConnector):
    class OpenAIModel(StrEnum):
        GPT_4 = "gpt-4"
        GPT_Turbo_4 = "gpt-4-turbo-2024-04-09" # gpt-4-turbo for latest
        GPT_Turbo_35 = "gpt-3.5-turbo"
        GPT_4O = "gpt-4o"
        GPT_4O_mini = "gpt-4o-mini"
    
    CLIENT = None
    TOTAL_TOKEN_COUNT = 0

    def get_client(self):
        if OpenAIChat.CLIENT is None:
            from auth import OpenAI_AUTH
            OpenAIChat.CLIENT = openai.OpenAI(api_key=OpenAI_AUTH)
        return OpenAIChat.CLIENT
    
    def __init__(self, openAI_model: OpenAIModel, system_message: str|None=None, chat_format=True):
        if openAI_model == OpenAIChat.OpenAIModel.GPT_4O:
            print("[Warning] Using the more expensive GPT 4O model")
        self.openAI_model = openAI_model
        self.chat_format = chat_format
        self.chat_log = [] if system_message is None else [{"role": LLMConnector.Role.System, "content": system_message}]
        super().__init__(str(self.openAI_model))
    
    def copy(self) -> OpenAIChat:
        ret = OpenAIChat(self.openAI_model, None, self.chat_format)
        ret.chat_log = copy.deepcopy(self.chat_log)
        return ret

    def dump(self, filename) -> None:
        with open(filename, 'w', encoding='utf8') as f:
            f.write(f"model: {self.openAI_model}\n")
            for message in self.chat_log:
                f.write(f"[{message['role']}]\n")
                f.write(f"{message['content']}\n")
    
    def _get_response(self, request) -> str:
        response = self.get_client().chat.completions.create(**request)
        tokens_used = int(response.usage.total_tokens) # for logging purposes
        OpenAIChat.TOTAL_TOKEN_COUNT += tokens_used
        response_text = response.choices[0].message.content
        if self.chat_format:
            self.chat_log = request['messages'] + [{'role': LLMConnector.Role.Assistant, 'content': response_text}]
        return response_text
    
    def inject(self, role: LLMConnector.Role, text: str):
        self.chat_log.append({"role": role, "content": text})
    
    def _prompt_to_request(self, prompt: str) -> object:
        # https://platform.openai.com/docs/guides/chat-completions/overview
        return {
                'model': self.openAI_model,
                'messages': self.chat_log + [{"role": LLMConnector.Role.User, "content": prompt}],
            }

    def prompt_as_API_request(self, prompt: str, request_id: str):
        request = self._prompt_to_request(prompt)
        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request,
        }