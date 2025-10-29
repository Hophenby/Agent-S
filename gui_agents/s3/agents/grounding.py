import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytesseract
from PIL import Image
from pytesseract import Output
import cv2
import numpy as np
import base64
from io import BytesIO

from memory.procedural_memory import PROCEDURAL_MEMORY
from core.mllm import LMMAgent
from utils.common_utils import call_llm_safe
from agents.code_agent import CodeAgent
import logging

logger = logging.getLogger("desktopenv.agent")

feedback_renderer: Optional[Callable[[Image.Image], bytes]] = None

def get_feedback_renderer() -> Optional[Callable[[Image.Image], bytes]]:
    return feedback_renderer
class ACI:
    def __init__(self):
        self.notes: List[str] = []


# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func


UBUNTU_APP_SETUP = f"""import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""


SET_CELL_VALUES_CMD = """import uno
import subprocess
import unicodedata, json

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"

    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"

    if component.supportsService("com.sun.star.sheet.PresentationDocument"):
        return "Impress"

    return None

def _norm_name(s: str | None) -> str | None:
    if s is None:
        return None
    if "\\\\u" in s or "\\\\U" in s or "\\\\x" in s:
        try:
            # json.loads handles all the escape forms safely
            s = json.loads(f"{{s}}")
        except Exception:
            # fallback: best-effort
            try:
                s = s.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass
    # Normalize (NFC works well across platforms)
    return unicodedata.normalize("NFC", s)

def cell_ref_to_indices(cell_ref):
    column_letters = ''.join(filter(str.isalpha, cell_ref))
    row_number = ''.join(filter(str.isdigit, cell_ref))

    col = sum((ord(char.upper()) - ord('A') + 1) * (26**idx) for idx, char in enumerate(reversed(column_letters))) - 1
    row = int(row_number) - 1
    return col, row

def set_cell_values(new_cell_values: dict[str, str], app_name: str = "Untitled 1", sheet_name: str = "Sheet1"):
    app_name  = _norm_name(app_name)
    sheet_name = _norm_name(sheet_name)

    new_cell_values_idx = {{}}
    for k, v in new_cell_values.items():
        try:
            col, row = cell_ref_to_indices(k)
        except:
            col = row = None

        if col is not None and row is not None:
            new_cell_values_idx[(col, row)] = v

    # Clean up previous TCP connections.
    subprocess.run(
        'echo \"osworld-public-evaluation\" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        check=True,
        text=True,
        capture_output=True
    )

    # Dynamically allow soffice to listen on port 2002.
    subprocess.run(
        [
            "soffice",
            "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"
        ]
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    context = resolver.resolve(
        f"uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Collect all LibreOffice-related opened windows.
    documents = []
    for i, component in enumerate(desktop.Components):
        title = component.Title
        doc_type = identify_document_type(component)
        documents.append((i, component, title, doc_type))

    # Find the LibreOffice Calc app and the sheet of interest.
    spreadsheet = [doc for doc in documents if doc[3] == "Calc"]
    selected_spreadsheet = [doc for doc in spreadsheet if doc[2] == app_name]
    if spreadsheet:
        try:
            if selected_spreadsheet:
                spreadsheet = selected_spreadsheet[0][1]
            else:
                spreadsheet = spreadsheet[0][1]

            sheet = spreadsheet.Sheets.getByName(sheet_name)
        except:
            raise ValueError(f"Could not find sheet {{sheet_name}} in {{app_name}}.")

        for (col, row), value in new_cell_values_idx.items():
            cell = sheet.getCellByPosition(col, row)

            # Set the cell value.
            if isinstance(value, (int, float)):
                cell.Value = value
            elif isinstance(value, str):
                if value.startswith("="):
                    cell.Formula = value
                else:
                    cell.String = value
            elif isinstance(value, bool):
                cell.Value = 1 if value else 0
            elif value is None:
                cell.clearContents(0)
            else:
                raise ValueError(f"Unsupported cell value type: {{type(value)}}")

    else:
        raise ValueError(f"Could not find LibreOffice Calc app corresponding to {{app_name}}.")

set_cell_values(new_cell_values={cell_values}, app_name="{app_name}", sheet_name="{sheet_name}")        
"""


# ACI primitives are parameterized by description, and coordinate generation uses a pretrained grounding model
class OSWorldACI(ACI):
    def __init__(
        self,
        env,
        platform: str,
        engine_params_for_generation: Dict,
        engine_params_for_grounding: Dict,
        width: int = 1920,
        height: int = 1080,
        code_agent_budget: int = 20,
        code_agent_engine_params: Dict = None,
    ):
        super().__init__()

        self.env = env
        self.platform = (
            platform  # Dictates how the switch_applications agent action works.
        )

        # Configure scaling
        self.width = width
        self.height = height

        # Maintain state for save_to_knowledge
        self.notes = []

        # Screenshot used during ACI execution
        self.obs = None

        # Configure the visual grounding model responsible for coordinate generation
        self.grounding_model = LMMAgent(engine_params_for_grounding)
        self.engine_params_for_grounding = engine_params_for_grounding

        # Configure text grounding agent
        self.text_span_agent = LMMAgent(
            engine_params=engine_params_for_generation,
            system_prompt=PROCEDURAL_MEMORY.PHRASE_TO_WORD_COORDS_PROMPT,
        )

        # Configure code agent
        code_agent_engine_params = (
            code_agent_engine_params or engine_params_for_generation
        )
        self.code_agent = CodeAgent(code_agent_engine_params, code_agent_budget)

        # Store task instruction for code agent
        self.current_task_instruction = None
        self.last_code_agent_result = None

    # Given the state and worker's referring expression, use the grounding model to generate (x,y)
    def generate_coords(self, ref_expr: str, obs: Dict) -> List[int]:

        # Reset the grounding model state
        self.grounding_model.reset()

        # Configure the context, UI-TARS demo does not use system prompt
        prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point in your response.\n"
        self.grounding_model.add_message(
            text_content=prompt, image_content=obs["screenshot"], put_text_last=True
        )

        # Generate and parse coordinates
        response = call_llm_safe(self.grounding_model)
        print("RAW GROUNDING MODEL RESPONSE:", response)
        numericals = re.findall(r"\d+", response)
        assert len(numericals) >= 2
        return [int(numericals[0]), int(numericals[1])]

    # Calls pytesseract to generate word level bounding boxes for text grounding
    def get_ocr_elements(self, b64_image_data: str) -> Tuple[str, List]:
        image = Image.open(BytesIO(b64_image_data))
        image_data = pytesseract.image_to_data(image, output_type=Output.DICT)

        # Clean text by removing leading and trailing spaces and non-alphabetical characters, but keeping punctuation
        for i, word in enumerate(image_data["text"]):
            image_data["text"][i] = re.sub(
                r"^[^a-zA-Z\s.,!?;:\-\+]+|[^a-zA-Z\s.,!?;:\-\+]+$", "", word
            )

        ocr_elements = []
        ocr_table = "Text Table:\nWord id\tText\n"
        # Obtain the <id, text, group number, word number> for each valid element
        grouping_map = defaultdict(list)
        ocr_id = 0
        for i in range(len(image_data["text"])):
            block_num = image_data["block_num"][i]
            if image_data["text"][i]:
                grouping_map[block_num].append(image_data["text"][i])
                ocr_table += f"{ocr_id}\t{image_data['text'][i]}\n"
                ocr_elements.append(
                    {
                        "id": ocr_id,
                        "text": image_data["text"][i],
                        "group_num": block_num,
                        "word_num": len(grouping_map[block_num]),
                        "left": image_data["left"][i],
                        "top": image_data["top"][i],
                        "width": image_data["width"][i],
                        "height": image_data["height"][i],
                    }
                )
                ocr_id += 1

        return ocr_table, ocr_elements

    # Given the state and worker's text phrase, generate the coords of the first/last word in the phrase
    def generate_text_coords(
        self, phrase: str, obs: Dict, alignment: str = ""
    ) -> List[int]:

        ocr_table, ocr_elements = self.get_ocr_elements(obs["screenshot"])

        alignment_prompt = ""
        if alignment == "start":
            alignment_prompt = "**Important**: Output the word id of the FIRST word in the provided phrase.\n"
        elif alignment == "end":
            alignment_prompt = "**Important**: Output the word id of the LAST word in the provided phrase.\n"

        # Load LLM prompt
        self.text_span_agent.reset()
        self.text_span_agent.add_message(
            alignment_prompt + "Phrase: " + phrase + "\n" + ocr_table, role="user"
        )
        self.text_span_agent.add_message(
            "Screenshot:\n", image_content=obs["screenshot"], role="user"
        )

        # Obtain the target element
        response = call_llm_safe(self.text_span_agent)
        print("TEXT SPAN AGENT RESPONSE:", response)
        numericals = re.findall(r"\d+", response)
        if len(numericals) > 0:
            text_id = int(numericals[-1])
        else:
            text_id = 0
        elem = ocr_elements[text_id]

        # Compute the element coordinates
        if alignment == "start":
            coords = [elem["left"], elem["top"] + (elem["height"] // 2)]
        elif alignment == "end":
            coords = [elem["left"] + elem["width"], elem["top"] + (elem["height"] // 2)]
        else:
            coords = [
                elem["left"] + (elem["width"] // 2),
                elem["top"] + (elem["height"] // 2),
            ]
        return coords

    def assign_screenshot(self, obs: Dict):
        self.obs = obs

    def set_task_instruction(self, task_instruction: str):
        """Set the current task instruction for the code agent."""
        self.current_task_instruction = task_instruction

    # Resize from grounding model dim into OSWorld dim (1920 * 1080)
    def resize_coordinates(self, coordinates: List[int]) -> List[int]:
        grounding_width = self.engine_params_for_grounding["grounding_width"]
        grounding_height = self.engine_params_for_grounding["grounding_height"]

        return [
            round(coordinates[0] * self.width / grounding_width),
            round(coordinates[1] * self.height / grounding_height),
        ]

    @agent_action
    def click(
        self,
        element_description: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """MCP 接口：根据元素描述执行点击（使用 grounding 模型生成坐标）。

        名称: OSWorldACI.click

        参数:
            element_description (str): 必需，对目标元素的自然语言描述（完整句子）。
            num_clicks (int): 可选，点击次数，默认 1。
            button_type (str): 可选，'left'|'middle'|'right'，默认 'left'。
            hold_keys (list): 可选，点击时需要按住的修饰按键列表。

        返回:
            str: 返回可执行的 pyautogui 命令字符串（由上层执行器决定是否执行）。

        错误:
            ValueError: 当 grounding 未能生成有效坐标时抛出。

        示例:
            {"method":"OSWorldACI.click","params":{"element_description":"点击搜索框"}}
        """
        coords1 = self.generate_coords(element_description, self.obs)
        x, y = self.resize_coordinates(coords1)
        command = "import pyautogui; "

        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"""import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); """
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        # Return pyautoguicode to click on the element
        return command

    @agent_action
    def switch_applications(self, app_code):
        """MCP 接口：切换到已经打开的应用程序（平台相关实现）。

        名称: OSWorldACI.switch_applications

        参数:
            app_code (str): 必需，目标应用的标识或名称（例如窗口标题或注册名称）。

        返回:
            str: 返回用于执行切换的命令字符串（平台相关）。

        错误:
            AssertionError: 当 platform 不被支持时抛出。
        """
        if self.platform == "darwin":
            return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"
        elif self.platform == "linux":
            return UBUNTU_APP_SETUP.replace("APP_NAME", app_code)
        elif self.platform == "windows":
            return f"import pyautogui; import time; pyautogui.hotkey('win', 'd', interval=0.5); pyautogui.typewrite({repr(app_code)}); pyautogui.press('enter'); time.sleep(1.0)"
        else:
            assert (
                False
            ), f"Unsupported platform: {self.platform}. Supported platforms are: darwin, linux, windows."

    @agent_action
    def open(self, app_or_filename: str):
        """MCP 接口：打开指定应用或文件。

        名称: OSWorldACI.open

        参数:
            app_or_filename (str): 必需，应用名或文件名，用于在桌面环境中打开目标。

        返回:
            str: 返回执行打开操作的命令字符串（平台相关）。

        错误:
            AssertionError: 当 platform 不支持时抛出。
        """
        if self.platform == "linux":
            return f"import pyautogui; pyautogui.hotkey('win'); time.sleep(0.5); pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); pyautogui.hotkey('enter'); time.sleep(0.5)"
        elif self.platform == "darwin":
            return f"import pyautogui; import time; pyautogui.hotkey('command', 'space', interval=0.5); pyautogui.typewrite({repr(app_or_filename)}); pyautogui.press('enter'); time.sleep(1.0)"
        elif self.platform == "windows":
            return (
                "import pyautogui; import time; "
                "pyautogui.hotkey('win'); time.sleep(0.5); "
                f"pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); "
                "pyautogui.press('enter'); time.sleep(0.5)"
            )
        else:
            assert (
                False
            ), f"Unsupported platform: {self.platform}. Supported platforms are: darwin, linux, windows."

    @agent_action
    def type(
        self,
        element_description: Optional[str] = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """MCP 接口：在指定元素中输入文本（可选先定位并点击元素）。

        名称: OSWorldACI.type

        参数:
            element_description (str|None): 可选，对目标元素的自然语言描述；若提供将先定位并点击。
            text (str): 要输入的文本（支持 Unicode）。
            overwrite (bool): 是否先清空元素再输入（True/False）。
            enter (bool): 是否在输入完成后按回车键。

        返回:
            str: 返回用于执行输入的命令字符串（pyautogui 或 clipboard 方法）。

        示例:
            {"method":"OSWorldACI.type","params":{"element_description":"搜索框","text":"hello","enter":true}}
        """
        command = "import pyautogui; "
        command += (
            "\ntry:\n"
            "    import pyperclip\n"
            "except ImportError:\n"
            "    import subprocess\n"
            "    subprocess.run('echo \"osworld-public-evaluation\" | sudo -S apt-get install -y xclip xsel', shell=True, check=True)\n"
            "    subprocess.check_call([subprocess.sys.executable, '-m', 'pip', 'install', 'pyperclip'])\n"
            "    import pyperclip\n\n"
        )

        if element_description is not None:
            coords1 = self.generate_coords(element_description, self.obs)
            x, y = self.resize_coordinates(coords1)
            command += f"pyautogui.click({x}, {y}); "

        if overwrite:
            command += (
                f"pyautogui.hotkey({repr('command' if self.platform == 'darwin' else 'ctrl')}, 'a'); "
                "pyautogui.press('backspace'); "
            )

        # Check if text contains Unicode characters that pyautogui.write() can't handle
        has_unicode = any(ord(char) > 127 for char in text)

        if has_unicode:
            # Use clipboard method for Unicode characters
            command += f"pyperclip.copy({repr(text)}); "
            command += f"pyautogui.hotkey({repr('command' if self.platform == 'darwin' else 'ctrl')}, 'v'); "
        else:
            # Use regular pyautogui.write() for ASCII text
            command += f"pyautogui.write({repr(text)}); "

        if enter:
            command += "pyautogui.press('enter'); "
        return command

    @agent_action
    def save_to_knowledge(self, text: List[str]):
        """MCP 接口：将若干文本保存到长期知识库以便任务期间重用（如复制粘贴、元素记忆）。

        名称: OSWorldACI.save_to_knowledge

        参数:
            text (List[str]): 必需，要保存的文本列表。

        返回:
            str: 返回 WAIT（占位），表示已记录内容供后续使用。
        """
        self.notes.extend(text)
        return """WAIT"""

    @agent_action
    def drag_and_drop(
        self, starting_description: str, ending_description: str, hold_keys: List = []
    ):
        """MCP 接口：根据起始/结束描述执行拖拽操作（使用 grounding 模型生成坐标）。

        名称: OSWorldACI.drag_and_drop

        参数:
            starting_description (str): 必需，起始元素的自然语言描述。
            ending_description (str): 必需，结束元素的自然语言描述。
            hold_keys (List): 可选，拖拽过程中需要按住的修饰键。

        返回:
            str: 返回用于执行拖拽的 pyautogui 命令字符串。

        错误:
            ValueError: grounding 过程中若无法定位到元素会抛出错误。
        """
        coords1 = self.generate_coords(starting_description, self.obs)
        coords2 = self.generate_coords(ending_description, self.obs)
        x1, y1 = self.resize_coordinates(coords1)
        x2, y2 = self.resize_coordinates(coords2)

        command = "import pyautogui; "

        command += f"pyautogui.moveTo({x1}, {y1}); "
        # TODO: specified duration?
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1., button='left'); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        # Return pyautoguicode to drag and drop the elements

        return command

    @agent_action
    def highlight_text_span(
        self, starting_phrase: str, ending_phrase: str, button: str = "left"
    ):
        """MCP 接口：根据起始/结束短语高亮文本段（用于词、行或段落）。

        名称: OSWorldACI.highlight_text_span

        参数:
            starting_phrase (str): 必需，标记高亮开始位置的短语。
            ending_phrase (str): 必需，标记高亮结束位置的短语。
            button (str): 可选，使用的鼠标按钮，'left'|'right'|'middle'，默认 'left'。

        返回:
            str: pyautogui 命令字符串，用于执行移动与拖拽完成高亮。
        """
        coords1 = self.generate_text_coords(
            starting_phrase, self.obs, alignment="start"
        )
        coords2 = self.generate_text_coords(ending_phrase, self.obs, alignment="end")
        x1, y1 = coords1
        x2, y2 = coords2

        command = "import pyautogui; "
        command += f"pyautogui.moveTo({x1}, {y1}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1., button='{button}'); pyautogui.mouseUp(); "

        # Return pyautoguicode to drag and drop the elements
        return command

    @agent_action
    def set_cell_values(
        self, cell_values: Dict[str, Any], app_name: str, sheet_name: str
    ):
        """MCP 接口：在表格应用中设置若干单元格的值（通过 CodeAgent 执行，返回的是动态生成的脚本）。

        名称: OSWorldACI.set_cell_values

        参数:
            cell_values (Dict[str, Any]): 必需，键为单元格坐标（如 "A1"），值为要设置的内容（支持数值/字符串/公式/布尔）。
            app_name (str): 必需，目标表格应用窗口/文档名称。
            sheet_name (str): 必需，要操作的工作表名称。

        返回:
            str: 返回一段会在目标环境中运行的脚本（LibreOffice UNO 操作），用于实际设置单元格。
        """
        return SET_CELL_VALUES_CMD.format(
            cell_values=cell_values, app_name=app_name, sheet_name=sheet_name
        )

    @agent_action
    def call_code_agent(self, task: str = None):
        """MCP 接口：调用 CodeAgent 来执行仅靠代码即可完成的子任务或特定任务。

        名称: OSWorldACI.call_code_agent

        参数:
            task (str|None): 可选。若为 None，则使用当前任务指令（self.current_task_instruction）；
                若为字符串，则表示要执行的明确子任务（仅在确知为子任务时使用）。

        返回:
            str: 返回执行器可运行的脚本/命令字符串（或 WAIT 占位），同时 CodeAgent 的执行结果会被存入 self.last_code_agent_result。

        注意 (重要):
            - 仅当 task 为明确子任务时传入 task 参数；不要为完整任务传入 task（以免造成 hallucination）。
            - 如果不确定，请传 None，让系统使用当前任务指令。

        示例:
            {"method":"OSWorldACI.call_code_agent","params":{"task":"Calculate sum of column B"}}
        """
        logger.info("=" * 50)
        logger.info("GROUNDING AGENT: Calling Code Agent")
        logger.info("=" * 50)

        # **CRITICAL**: Only use provided task for specific subtasks, otherwise use original task instruction
        if task is not None:
            # This is a subtask - use the provided task
            task_to_execute = task
            logger.info(f"Executing SUBTASK: {task_to_execute}")
        else:
            # This is a full task - use the original task instruction to prevent hallucination
            task_to_execute = self.current_task_instruction
            logger.info(f"Executing FULL TASK: {task_to_execute}")

        if task_to_execute:
            print("obs keys: ", self.obs.keys())
            screenshot = self.obs.get("screenshot", "") if self.obs else ""
            logger.info(f"Screenshot available: {'Yes' if screenshot else 'No'}")

            logger.info("Executing code agent...")
            result = self.code_agent.execute(
                task_to_execute, screenshot, self.env.controller
            )

            # Store the result for the worker to access
            self.last_code_agent_result = result

            logger.info("Code agent execution completed")
            logger.info(f"Result - Completion reason: {result['completion_reason']}")
            logger.info(f"Steps executed: {result['steps_executed']}")
            logger.info(f"Summary: {result['summary']}")

            logger.info("=" * 50)
            logger.info("GROUNDING AGENT: Code Agent Call Finished")
            logger.info("=" * 50)

            # Return code to be executed in the environment
            return "import time; time.sleep(2.222)"
        else:
            logger.warning("No task instruction available for code agent call")
            return "import time; time.sleep(1.111)"

    @agent_action
    def scroll(self, element_description: str, clicks: int, shift: bool = False):
        """MCP 接口：对指定元素执行滚轮滚动（使用 grounding 模型定位元素）。

        名称: OSWorldACI.scroll

        参数:
            element_description (str): 必需，对目标元素的自然语言描述。
            clicks (int): 必需，滚动步数（正/负表示方向）。
            shift (bool): 可选，若为 True 则使用水平滚动（shift+滚轮），否则垂直滚动。

        返回:
            str: 返回执行滚动的命令字符串（pyautogui）。
        """
        coords1 = self.generate_coords(element_description, self.obs)
        x, y = self.resize_coordinates(coords1)

        if shift:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.hscroll({clicks})"
        else:
            return f"import pyautogui; import time; pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.vscroll({clicks})"

    @agent_action
    def hotkey(self, keys: List):
        """MCP 接口：按下组合键。

        名称: OSWorldACI.hotkey

        参数:
            keys (List[str]): 必需，表示要按下的键序列，例如 ['ctrl','c']。

        返回:
            str: 返回可执行的 pyautogui.hotkey 命令字符串。
        """
        # add quotes around the keys
        keys = [f"'{key}'" for key in keys]
        return f"import pyautogui; pyautogui.hotkey({', '.join(keys)})"

    @agent_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """MCP 接口：按住一组按键后顺序按下另一组按键。

        名称: OSWorldACI.hold_and_press

        参数:
            hold_keys (List[str]): 必需，要按住的按键列表。
            press_keys (List[str]): 必需，要依次按下的按键列表。

        返回:
            str: 返回用于在环境中执行的 pyautogui 命令字符串。
        """

        press_keys_str = "[" + ", ".join([f"'{key}'" for key in press_keys]) + "]"
        command = "import pyautogui; "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.press({press_keys_str}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        return command

    @agent_action
    def wait(self, time: float):
        """Wait for a specified amount of time
        Args:
            time:float the amount of time to wait in seconds
        """
        return f"""import time; time.sleep({time})"""

    @agent_action
    def done(
        self,
    ):
        """End the current task with a success. Use this when you believe the entire task has been fully completed."""
        return """DONE"""

    @agent_action
    def fail(self):
        """End the current task with a failure. Use this when you believe the entire task is impossible to complete."""
        return """FAIL"""


class LegacyACI(ACI):
    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__()
        # default screen size (can be overridden)
        self.width = width
        self.height = height
        # last operation info for feedback drawing
        self._last_op = None
        # assignable screenshot (raw bytes or PIL image)
        self.obs = None

    def assign_screenshot(self, obs: Dict):
        """MCP 接口：分配当前截图（供后续操作和反馈绘制使用）。

        名称: LegacyACI.assign_screenshot

        参数:
            obs (dict): 必需，包含键 'screenshot' 的字典。screenshot 支持以下格式：
                - base64 编码的 JPEG/PNG 字符串
                - 原始 image bytes
                - PIL.Image 对象
                - numpy.ndarray

        返回:
            dict: {"result": "OK"}（在本地实现中将直接设置并无复杂返回值）

        错误:
            ValueError: 当提供的图片无法解析时可抛出 invalid_image 错误（调用者可捕获并处理）。

        示例:
            agent.assign_screenshot({"screenshot": "<base64-string>"})
        """
        self.obs = obs

    def _to_abs_point(self, rel):
        """Convert relative coordinate or bbox to absolute point (x,y).
        Accepts:
          - point: (x_rel, y_rel) in [0,1]
          - bbox: (x_rel, y_rel, w_rel, h_rel) where x_rel,y_rel are top-left
        Returns (x,y) integers in pixels.
        """
        if rel is None:
            return None
        if len(rel) == 2:
            x = int(round(rel[0] * self.width))
            y = int(round(rel[1] * self.height))
            return x, y
        elif len(rel) >= 4:
            x = int(round((rel[0] + rel[2] / 2.0) * self.width))
            y = int(round((rel[1] + rel[3] / 2.0) * self.height))
            return x, y
        else:
            raise ValueError("Relative coordinate must be len 2 or >=4")

    def _to_abs_bbox(self, rel):
        """Convert relative bbox (x,y,w,h) to absolute bbox in pixels: (x1,y1,x2,y2)"""
        if rel is None:
            return None
        if len(rel) >= 4:
            x1 = int(round(rel[0] * self.width))
            y1 = int(round(rel[1] * self.height))
            w = int(round(rel[2] * self.width))
            h = int(round(rel[3] * self.height))
            x2 = x1 + w
            y2 = y1 + h
            return x1, y1, x2, y2
        return None

    def _load_image_as_bgr(self):
        """Load screenshot from self.obs into a BGR numpy array for cv2 drawing.
        Returns a copy of the image sized to (height,width).
        """
        if not self.obs:
            # blank canvas
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255
            return img

        screenshot = self.obs.get("screenshot") if isinstance(self.obs, dict) else self.obs
        # if base64 string
        if isinstance(screenshot, str):
            try:
                b = base64.b64decode(screenshot)
            except Exception:
                b = screenshot.encode()
            screenshot = b

        # if bytes
        if isinstance(screenshot, (bytes, bytearray)):
            pil = Image.open(BytesIO(screenshot)).convert("RGB")
            arr = np.array(pil)
            # PIL gives RGB, convert to BGR
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif isinstance(screenshot, Image.Image):
            arr = np.array(screenshot.convert("RGB"))
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif isinstance(screenshot, np.ndarray):
            img = screenshot.copy()
            # If RGB, convert to BGR heuristically
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

        # Resize/crop/pad to target dims
        img_h, img_w = img.shape[:2]
        if (img_w, img_h) != (self.width, self.height):
            img = cv2.resize(img, (self.width, self.height))
        return img

    def _encode_img_to_b64(self, img_bgr) -> str:
        _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buf.tobytes()).decode()

    def _draw_feedback(self, coords_rel=None, bbox_rel=None, text: str = '') -> str:
        """Draw the last operation (point or bbox) and annotation text on the screenshot and return base64 jpg string."""
        img = self._load_image_as_bgr()
        # draw bbox if provided
        if bbox_rel is not None:
            bb = self._to_abs_bbox(bbox_rel)
            if bb:
                x1, y1, x2, y2 = bb
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif coords_rel is not None:
            x, y = self._to_abs_point(coords_rel)
            cv2.circle(img, (x, y), 12, (0, 0, 255), -1)
            cv2.putText(img, text, (max(10, x - 10), max(20, y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)

        return self._encode_img_to_b64(img)

    def _wrap_result(self, result: Any, coords_rel=None, bbox_rel=None, text: str = '') -> Dict:
        """Return a dict containing the execution result, a base64-encoded feedback image, and annotation text."""
        global feedback_renderer
        img_b64 = self._draw_feedback(coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)

        # Publish a renderer callable to the module-level `feedback_renderer` so external
        # callers (e.g., cli_app) can render this same annotation on arbitrary screenshots.
        def _renderer(screenshot_input) -> Image.Image:
            """Render the same feedback (coords_rel/bbox_rel/text) onto the provided screenshot.

            Accepts PIL.Image, raw bytes, numpy array, or dict{'screenshot': ...}.
            Returns a PIL.Image with the annotation drawn.
            """

            # Use the instance helper to generate JPEG bytes with annotation
            try:
                jpeg_bytes = self.draw_feedback_bytes(screenshot_input, coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)
                return jpeg_bytes
            except Exception:
                # Fallback: attempt to draw by temporarily setting obs and using _load_image_as_bgr/_draw_feedback
                prev_obs = getattr(self, "obs", None)
                try:
                    if isinstance(screenshot_input, dict):
                        self.obs = screenshot_input
                    else:
                        self.obs = {"screenshot": screenshot_input}
                    b64 = self._draw_feedback(coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)
                    return BytesIO(base64.b64decode(b64)).getvalue()
                finally:
                    try:
                        self.obs = prev_obs
                    except Exception:
                        self.obs = None

        # assign to module-level variable so cli_app can import and call it
        try:
            feedback_renderer = _renderer
            print("Assigned feedback_renderer callable for external use.")
        except Exception:
            pass

        return {
            "result": result,
            "feedback_image_base64": img_b64,
            "annotation": text,
        }

    # Public helpers to render feedback on an arbitrary screenshot (no side-effects)
    def draw_feedback_b64(self, screenshot_input, coords_rel=None, bbox_rel=None, text: str = "") -> str:
        """Render feedback (circle/bbox + text) on a given screenshot and return a base64 JPEG string.

        Args:
            screenshot_input: one of the supported screenshot formats (bytes, PIL.Image, numpy.ndarray, or a dict {'screenshot': ...}).
            coords_rel: optional relative point (x_rel,y_rel) in [0,1]
            bbox_rel: optional relative bbox (x,y,w,h) in [0,1]
            text: annotation text to draw near the operation

        Returns:
            base64-encoded JPEG string of the annotated image.

        Notes:
            - This method temporarily sets `self.obs` to the provided screenshot for rendering and restores the previous value on return.
            - It does not change any other state (best-effort).
        """
        prev_obs = getattr(self, "obs", None)
        try:
            # Accept either raw image payload or a wrapped dict
            if isinstance(screenshot_input, dict):
                self.obs = screenshot_input
            else:
                self.obs = {"screenshot": screenshot_input}

            return self._draw_feedback(coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)
        finally:
            # restore previous obs
            try:
                self.obs = prev_obs
            except Exception:
                self.obs = None

    def draw_feedback_bytes(self, screenshot_input, coords_rel=None, bbox_rel=None, text: str = "") -> bytes:
        """Render feedback and return raw JPEG bytes.

        Convenience wrapper that decodes the base64 string produced by draw_feedback_b64.
        """
        b64 = self.draw_feedback_b64(screenshot_input, coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)
        return base64.b64decode(b64)

    # Common actions using relative coordinates (values in [0,1])
    @agent_action
    def click(self, rel_point: Tuple[float, float], num_clicks: int = 1, button_type: str = "left"):
        """MCP 接口：在相对坐标点执行点击。

        名称: LegacyACI.click

        参数:
            rel_point (tuple[float, float]): 必需，相对坐标 [x_rel, y_rel]，范围 0.0-1.0。
            num_clicks (int): 可选，点击次数，默认 1。
            button_type (str): 可选，按钮类型，'left'|'middle'|'right'，默认 'left'

        返回:
            dict: 包含以下字段的字典：
                - result (str): 可执行命令字符串（例如 pyautogui 调用）。
                - feedback_image_base64 (str): base64 编码的 JPEG，图像上标注了点击点和文本注释。
                - annotation (str): 简短文本描述，例如 "Clicked at (960,540)"。

        错误:
            ValueError: 当 rel_point 坐标不在 [0,1] 范围内时可抛出 invalid_coordinate。

        示例请求 (JSON-RPC):
            {"method": "LegacyACI.click", "params": {"rel_point": [0.5,0.5], "num_clicks": 1}}
        """
        x_abs, y_abs = self._to_abs_point(rel_point)
        cmd = f"import pyautogui; pyautogui.click({x_abs}, {y_abs}, clicks={num_clicks}, button={repr(button_type)})"
        text = f"clicked here last time"
        return self._wrap_result(cmd, coords_rel=rel_point, text=text)

    @agent_action
    def drag_and_drop(self, rel_start: Tuple[float, float], rel_end: Tuple[float, float], duration: float = 1.0):
        """MCP 接口：从起始相对点拖拽到结束相对点。

        名称: LegacyACI.drag_and_drop

        参数:
            rel_start (tuple[float,float]): 必需，起始相对坐标 [x_rel,y_rel]。
            rel_end (tuple[float,float]): 必需，结束相对坐标 [x_rel,y_rel]。
            duration (float): 可选，拖拽时长（秒），默认 1.0。

        返回:
            dict: {result, feedback_image_base64, annotation}，其中 feedback_image_base64 的图像会绘制覆盖拖拽区域的矩形。

        错误:
            ValueError: 当坐标不在 [0,1] 范围内时抛出 invalid_coordinate。

        示例:
            {"method":"LegacyACI.drag_and_drop","params":{"rel_start":[0.2,0.2],"rel_end":[0.8,0.5]}}
        """
        x1, y1 = self._to_abs_point(rel_start)
        x2, y2 = self._to_abs_point(rel_end)
        cmd = f"import pyautogui; pyautogui.moveTo({x1},{y1}); pyautogui.dragTo({x2},{y2}, duration={duration}, button='left'); pyautogui.mouseUp()"
        text = f"Dragged from ({x1},{y1}) to ({x2},{y2})"
        # create bbox covering the drag line
        bbox_rel = (min(rel_start[0], rel_end[0]), min(rel_start[1], rel_end[1]), abs(rel_end[0]-rel_start[0]), abs(rel_end[1]-rel_start[1]))
        return self._wrap_result(cmd, bbox_rel=bbox_rel, text=text)

    @agent_action
    def type(self, rel_point: Optional[Tuple[float, float]] = None, text_to_type: str = "", enter: bool = False):
        """MCP 接口：在可选相对位置点击后输入文本，并可选择回车。

        名称: LegacyACI.type

        参数:
            rel_point (tuple|None): 可选，相对坐标点；若提供会先点击该点。
            text_to_type (str): 要输入的文本（可包含 Unicode）。
            enter (bool): 是否在输入后按回车。

        返回:
            dict: 包含 result（命令字符串）、feedback_image_base64（标注点击点）和 annotation（已输入文本的简短描述）。

        示例:
            {"method":"LegacyACI.type","params":{"rel_point":[0.5,0.5],"text_to_type":"Hello","enter":true}}
        """
        cmd = "import pyautogui; "
        if rel_point is not None:
            x, y = self._to_abs_point(rel_point)
            cmd += f"pyautogui.click({x},{y}); "
        # naive handling for unicode: use clipboard approach would complicate; keep simple
        cmd += f"pyautogui.write({repr(text_to_type)}); "
        if enter:
            cmd += "pyautogui.press('enter'); "
        ann = f"Typed text: {text_to_type[:80]}"
        return self._wrap_result(cmd, coords_rel=rel_point, text=ann)

    @agent_action
    def scroll(self, rel_point: Tuple[float, float], clicks: int, horizontal: bool = False):
        """MCP 接口：在相对坐标处执行滚轮滚动。

        名称: LegacyACI.scroll

        参数:
            rel_point (tuple[float,float]): 必需，相对坐标点。
            clicks (int): 必需，滚动步数。正数/负数表示方向（依平台约定）。
            horizontal (bool): 可选，是否水平滚动，默认为 False（垂直）。

        返回:
            dict: 包含 result（命令字符串）、feedback_image_base64（标注滚动点）和 annotation（描述）。
        """
        x, y = self._to_abs_point(rel_point)
        if horizontal:
            cmd = f"import pyautogui; pyautogui.moveTo({x},{y}); pyautogui.hscroll({clicks})"
        else:
            cmd = f"import pyautogui; pyautogui.moveTo({x},{y}); pyautogui.vscroll({clicks})"
        ann = f"Scrolled at ({x},{y}) by {clicks}"
        return self._wrap_result(cmd, coords_rel=rel_point, text=ann)

    @agent_action
    def hotkey(self, keys: List[str]):
        """MCP 接口：按下组合键。

        名称: LegacyACI.hotkey

        参数:
            keys (list[str]): 必需，表示按键序列，例如 ["ctrl","c"]。

        返回:
            dict: {result, feedback_image_base64, annotation}。annotation 示例: "Hotkey pressed: ctrl+c"。

        示例:
            {"method":"LegacyACI.hotkey","params":{"keys":["ctrl","s"]}}
        """
        keys_quoted = [f"'{k}'" for k in keys]
        cmd = f"import pyautogui; pyautogui.hotkey({', '.join(keys_quoted)})"
        ann = f"Hotkey pressed: {'+'.join(keys)}"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def wait(self, seconds: float):
        """MCP 接口：等待指定秒数。

        名称: LegacyACI.wait

        参数:
            seconds (float): 必需，等待时长（秒），必须 >= 0。

        返回:
            dict: {result, feedback_image_base64, annotation}。
        """
        cmd = f"import time; time.sleep({seconds})"
        ann = f"Waited for {seconds} seconds"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def switch_applications(self, app_code: str):
        """MCP 接口：按名称切换到已有应用（占位实现）。

        名称: LegacyACI.switch_applications

        参数:
            app_code (str): 必需，目标应用的标识或名称。

        返回:
            dict: {result, feedback_image_base64, annotation}。result 为占位命令字符串，annotation 为简短描述。
        """
        # best-effort: no coordinate used, return simple annotation
        cmd = f"# switch_applications placeholder for {app_code}"
        ann = f"Switch to app: {app_code}"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def open(self, app_or_filename: str):
        """MCP 接口：打开应用或文件（占位）。

        名称: LegacyACI.open

        参数:
            app_or_filename (str): 必需，应用名或文件名。

        返回:
            dict: {result, feedback_image_base64, annotation}。
        """
        cmd = f"# open placeholder for {app_or_filename}"
        ann = f"Open: {app_or_filename}"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def done(self):
        """MCP 接口：标记任务完成（成功）。

        名称: LegacyACI.done

        参数: 无

        返回:
            dict: {result: "DONE", feedback_image_base64, annotation}
        """
        return self._wrap_result("DONE", text="Task completed")

    @agent_action
    def fail(self):
        """MCP 接口：标记任务失败。

        名称: LegacyACI.fail

        参数: 无

        返回:
            dict: {result: "FAIL", feedback_image_base64, annotation}
        """
        return self._wrap_result("FAIL", text="Task failed")