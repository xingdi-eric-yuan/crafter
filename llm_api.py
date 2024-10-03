import os
import json
import logging

from termcolor import colored
from openai import OpenAI, AzureOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential


logger = logging.getLogger("auto-debug")

LLM_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "llm.cfg")


if os.path.exists(LLM_CONFIG_FILE):
    LLM_CONFIGS = json.load(open(LLM_CONFIG_FILE))


def is_rate_limit_error(exception):
    # List of fully qualified names of RateLimitError exceptions from various libraries
    rate_limit_errors = [
        "openai.APIStatusError",
        "openai.APITimeoutError",
        "openai.error.Timeout",
        "openai.error.RateLimitError",
        "openai.error.ServiceUnavailableError",
        "openai.Timeout",
        "openai.APIError",
        "openai.APIConnectionError",
        "openai.RateLimitError",
        # Add more as needed
    ]
    exception_full_name = f"{exception.__class__.__module__}.{exception.__class__.__name__}"
    logger.warning(f"Exception_full_name: {exception_full_name}")
    logger.warning(f"Exception: {exception}")
    return exception_full_name in rate_limit_errors


class LLM:
    def __init__(self, model_name, verbose=False):
        self.model_name = model_name
        self.verbose = verbose

        if self.model_name not in LLM_CONFIGS:
            raise Exception(f"Model {self.model_name} not found in llm.cfg")

        self.configs = LLM_CONFIGS[self.model_name]
        self.context_length = self.configs["context_limit"] * 1000
        print(f"Using {self.model_name} with max context length of {self.context_length:,} tokens.")

        if "azure openai" in self.configs.get("tags", []):
            self.client = AzureOpenAI(api_key=self.configs["api_key"], azure_endpoint=self.configs["endpoint"], api_version=self.configs["api_version"], timeout=None)
        else:
            self.client = OpenAI(api_key=self.configs["api_key"], base_url=self.configs["endpoint"], timeout=None)

    @retry(
        retry=retry_if_exception(is_rate_limit_error),
        wait=wait_random_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(100),
    )
    def query_model(self, messages, json_format=False, **kwargs):
        kwargs["max_tokens"] = kwargs.get("max_tokens", self.configs.get("max_tokens"))

        return self.client.chat.completions.create(
            model=self.configs["model"],
            messages=messages,
            response_format={"type": "json_object" if json_format else "text"},
            **kwargs,
        ).choices[0].message.content

    def __call__(self, messages, json_format=False, *args, **kwargs):
        if not self.configs.get("system_prompt_support", True):
            # Replace system by user
            for i, m in enumerate(messages):
                if m["role"] == "system":
                    messages[i]["role"] = "user"

        # Merge consecutive messages with same role.
        messages = self.merge_messages(messages)

        # Trim message content to context length
        for i, m in enumerate(messages):
            messages[i]["content"] = messages[i]["content"][-self.context_length:]

        if self.verbose:
            # Message is a list of dictionaries with role and content keys.
            # Color each role differently.
            for m in messages:
                if m["role"] == "user":
                    print(colored(f"{m['content']}\n", "cyan"))
                elif m["role"] == "assistant":
                    print(colored(f"{m['content']}\n", "green"))
                elif m["role"] == "system":
                    print(colored(f"{m['content']}\n", "yellow"))
                else:
                    raise ValueError(f"Unknown role: {m['content']}")

        response = self.query_model(messages, json_format, **kwargs)
        response = response.strip()
        if response.startswith("```python") and response.endswith("```"):
            response = response[10:-3]
        elif response.startswith("```pdb") and response.endswith("```"):
            response = response[7:-3]
        elif response.startswith("```plaintext") and response.endswith("```"):
            response = response[13:-3]
        elif response.startswith("```bash") and response.endswith("```"):
            response = response[8:-3]
        elif response.startswith("```text") and response.endswith("```"):
            response = response[8:-3]
        elif response.startswith("```python"):
            response = response[10:]
            # find ``` at the end
            end_index = response.rfind("```")
            if end_index != -1:
                response = response[:end_index]
        elif response.startswith("```pdb"):
            response = response[7:]
            # find ``` at the end
            end_index = response.rfind("```")
            if end_index != -1:
                response = response[:end_index]
        elif response.endswith("```"):
            response = response[:-3]

        if self.verbose:
            print(colored(response, "green"))

        return response

    def merge_messages(self, messages):
        messages_out = [dict(messages[0])]
        for message in messages[1:]:
            if message["role"] == messages_out[-1]["role"]:
                messages_out[-1]["content"] += "\n\n" + message["content"]
            else:
                messages_out.append(dict(message))

        return messages_out
