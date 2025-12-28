from dataclasses import dataclass
from app.chat import ChatGenerator
import pandas as pd
import os
import subprocess
import sys
import json
from typing import List, Dict
import pickle
import re
import io
from app.common_agent import Tool, ToolResult
import logging

logging.basicConfig(
    filename="output.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class CodeWriter(Tool):
    def __init__(self, generator: ChatGenerator):
        super().__init__(
            name="code_writer",
            func="""
                Write Python code to temp.py and then execute it
                
                Arguments:
                `task`:
                - A single, concrete subtask that advances the main goal. It could be a subtask that help to move towards the goal.
                - Must be achievable in 50 lines of Python code.
                - Must NOT be ambiguous or require assumptions beyond what is provided in `context`.

                `context`:
                - A COMPLETE, SELF-SUFFICIENT specification required to implement `task`.
                - This is the ONLY information available to the code writer on this task.
            """
        )
        SYSTEM_PROMPT = """
            You are a code generation service.

            Your ONLY job is to write valid Python code.
            The following python package are available for you to use but other libraries are NOT so limit your use case to these ones:
            - matplotlib
            - numpy
            - sklearn
            - pandas
            - torch

            Rules:
            - Output Python code ONLY.
            - Do NOT include markdown.
            - Do NOT include explanations.
            - Do NOT include ```python!
            - The code must be executable as-is.
            - stdout will be kept as the output of the tool for future use. However, if there is any plotting / images required, make sure to output it to data/* folder
            - If you need to return a Python object, put it according to `output_artifact_path` to be used by future agents. 
                Take note that `output_artifact_path` does not refer to an input but rather the output of this program
            - If you are saving to `output_artifact_path`, print a VERY DETAILED description of the object to stdout so that the next program know what it is including variable name of objects and a full schema
        """

        self.generator = generator
        self.system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
        self.previous_output = ""

    def extract_code(self, text: str) -> str:
        text = text.strip()

        # Remove fenced code blocks of any language
        if text.startswith("```"):
            # Remove opening fence (``` or ```python etc.)
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            # Remove closing fence
            text = re.sub(r"\n?```$", "", text)

        text = text.strip()

        return text

    def execute(self, task: str, context: str) -> ToolResult:
        messages = [self.system_message]
        artifact_path = self.get_artifact_path()
        messages.append({
            "role": "user",
            "content": f"""
                Task:\n{task}\n\n
                Context:\n{context}\n\n
                Previous Code Output:\n{self.previous_output}\n\n
                output_artifact_path: {artifact_path}
            """
        })

        _, code = self.generator.generate(messages)
        print(code)

        logging.info("code writer" + str(messages))

        with open("temp.py", "w") as f:
            f.write(self.extract_code(code))

        human_in_the_loop = input("Press Y to execute code")
        if human_in_the_loop != "Y":
            return ToolResult(
                artifact_path=None,
                result={
                    "success": False,
                    "stdout": "",
                    "stderr": "Rejected by Human"
                }
            )

        result = subprocess.run(
            [sys.executable, "temp.py"],
            capture_output=True,
            text=True
        )

        if result.stderr:
            print(f"ERROR: {result.stderr}")

        res = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "artifact_path": artifact_path
        }

        self.previous_output += f"\n{res}\n"

        return ToolResult(
            artifact_path=artifact_path, 
            result=res
        )


class DataReaderTool(Tool):
    def __init__(self):
        super().__init__(
            name="data_reader",
            func="Read a CSV file. Argument is `path`: the path of the file to read. Artifact is the data itself"
        )

    def execute(self, path) -> ToolResult:
        df = pd.read_csv(path)

        # Capture df.info() output
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        artifact_path = self.get_artifact_path()
        with open(artifact_path, "wb") as f:
            pickle.dump(df, f)

        return ToolResult(
            artifact_path=artifact_path,
            result={
                "info": info_str,
                "columns": df.dtypes.astype(str).to_dict(),
                "rows": len(df)
            }
        )
    
