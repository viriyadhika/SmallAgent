from dataclasses import dataclass
import json
from app.chat import ChatGenerator
from typing import List
import re
import time
import logging

logging.basicConfig(
    filename="output.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class ToolResult:
    artifact_path: str | None
    result: dict

class Tool:
    def __init__(self, name: str, func: str):
        self.name = name
        self.func = func
        self.artifact_id = 0

    def get_artifact_path(self):
        self.artifact_id += 1
        return "data/" + self.name + str(self.artifact_id) + ".pkl"

    def execute(self, *args, **kwargs) -> ToolResult:
        raise NotImplementedError
    
class Router:
    def __init__(self, generator: ChatGenerator, tools: List[Tool], verbose=False):
        all_tool_desc = [f"- {tool.name}: {tool.func}" for tool in tools]
        SYSTEM_PROMPT = """
            You are an autonomous agent.
            Your job is to decide the next action to take in order to achieve the user's goal. You have access to the following tools: 
            """  + \
            "\n".join(all_tool_desc) + \
            """

            Rules:
            - Decide ONE action at a time.
            - If a tool is needed, choose action = "tool".
            - If the task is complete, choose action = "finish".
            - You MUST respond in valid JSON only. That is able to be loaded by json.loads. Do NOT put new line character as it will cause JSON parse error
            - Do NOT include markdown ```json
            - Do NOT include explanations or extra text.
            - artifact_location is a pickle path that can be read and referred to for future code execution

            JSON schema:
            {
              "action": "tool" | "finish",
              "tool": "<tool_name>",
              "args": { "<arg_name>": "<value>" },
              "final_answer": "<only if action=finish>"
            }
        """
        self.generator = generator
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.verbose = verbose

    def add_user_goal(self, question: str):
        self.messages.append({
            "role": "user",
            "content": f"Goal:\n{question}"
        })

    def add_tool_result(self, tool_name: str, result: dict, artifact_path: str | None):
        self.messages.append({
            "role": "user",
            "content": "Tool" + f"`{tool_name}`" + "returned:\n" + f"{json.dumps(result)}" + "artifact_location: \n" + f"{str(artifact_path)}" \
            """ What do I do next? JSON schema (the output should be valid JSON that can be read by JSON.loads):
                {
                  "action": "tool" | "finish",
                  "tool": "<tool_name>",
                  "args": { "<arg_name>": "<value>" },
                  "final_answer": "<only if action=finish>"
                }
            """
        })

    def next_action(self):
        if self.verbose:
            print(self.messages)
        else:
            logging.info(self.messages)

        thinking, response = self.generator.generate(self.messages)
        self.messages.append({
            "role": "assistant",
            "content": response
        })
        if self.verbose:
            print(response)
        else:
            logging.info(response)
        return response

class Agent:
    def __init__(self, router: Router, tools: List[Tool], max_steps=10):
        self.router = router
        self.tools = {
            tool.name: tool for tool in tools
        }
        self.max_steps = max_steps


    def extract_json(self, text: str) -> dict:
        """
        Extract and parse JSON from a string.
        Handles:
        - raw JSON
        - ```json ... ```
        - ``` ... ```
        """
        text = text.strip()

        # Case 1: fenced code block
        if text.startswith("```"):
            # Remove opening and closing fences
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        return json.loads(text)

    def run(self, question: str):
        self.router.add_user_goal(question)

        for step in range(self.max_steps):
            response = self.router.next_action()

            try:
                command = self.extract_json(response)
            except json.JSONDecodeError:
                raise ValueError("Model did not return valid JSON")

            if command["action"] == "finish":
                return command["final_answer"]

            tool_name = command["tool"]

            args = command.get("args", {})
            tool = self.tools.get(tool_name)
            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")

            print(f"Executing Tool: {tool_name} with Args {args}")
            logging.info(f"Executing Tool: {tool_name} with Args {args}")
            result = tool.execute(**args)
            print(f"Tool Result: {result}")
            logging.info(f"Tool Result: {result}")

            # Store tool result in conversation memory
            self.router.add_tool_result(tool_name, result.result, result.artifact_path)
            time.sleep(2)

        raise RuntimeError("Agent did not finish within max_steps")
