import logging
import json
from typing import List, Dict, Any
from pathlib import Path

from .optimizer import OptimizerConfig, Optimizer
from ..agents.llm import LLMConfig, OpenAILLM
from ..evaluation import Case
from ..task import Solution
from ..agents.toolkit import Toolkit
from .utils import OptimUtils
from . import prompt_formatter


class ToolkitOptimizerConfig(OptimizerConfig):
    def __init__(self, config_path_or_dict):
        super().__init__(config_path_or_dict)

        self.max_actions_per_step = self.toolkit_optimizer_setting_dict.get("max_actions_per_step", 3)
        self.llm_config = self.toolkit_optimizer_setting_dict.get("llm_config", None)
        self.meta_prompt = self.toolkit_optimizer_setting_dict.get("meta_prompt", {})


class ToolkitOptimizer(Optimizer):
    def __init__(self, config: ToolkitOptimizerConfig, logger_name: str = None):
        super().__init__(config)
        self.config = config
        self.max_actions_per_step = self.config.max_actions_per_step

        llm_config = LLMConfig(self.config.llm_config) if self.config.llm_config else None
        self.llm = OpenAILLM(llm_config) if llm_config else None

        self.meta_backward = self.config.meta_prompt.get("backward", {})
        self.meta_optim = self.config.meta_prompt.get("optim", {})

        # logger
        self.logger = (
            logging.getLogger(logger_name)
            if logger_name
            else logging.getLogger(__name__)
        )

    def optimize(self, case_list: List[Case], solution: Solution, save_path: Path=None):
        shared_toolkit: Toolkit = solution.agent_team.environment.shared_toolkit
        if not shared_toolkit:
            self.logger.info("No shared toolkit found, skipping toolkit optimization.")
            return solution, False

        self.logger.info("Start Toolkit Optimization")

        if save_path is None:
            # Fallback if save_path is not provided, though Trainer typically provides it
            self.logger.warning("save_path not provided for ToolkitOptimizer.optimize, skipping detailed dump.")

        # 1. Backward pass: Analyze tool usage in each case
        for case in case_list:
             backward_save_dir = save_path / "backward" if save_path else None
             self.backward(case, solution, backward_save_dir)

        # 2. Optimization pass: Aggregate analysis and suggest changes
        # We need to construct a prompt that includes the toolkit config and the analysis from cases

        prompt = self.formulate_optim_prompt(solution, case_list)

        _, content = self.llm.get_response(
            chat_messages=None,
            system_prompt="",
            last_prompt=prompt,
            stream=False
        )

        # 3. Extract and Apply changes
        extracted_dict = OptimUtils.extract_data_from_response(content, self.meta_optim.get("extract_key", ["result"]))
        result = extracted_dict.get("result")

        op_status = False
        if result:
            try:
                op_list = json.loads(result)
                op_status = self.apply_changes(solution, op_list)
            except Exception as e:
                self.logger.error(f"Failed to apply toolkit optimization: {e}")
                self.logger.error(f"Result was: {result}")

        # Save optimization info
        if save_path:
            optim_info = {
                "optim_status": op_status,
                "result": result,
                "prompt": prompt,
                "response": content,
            }
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)

            with open(save_path / "toolkit_optim_info.json", "w", encoding="utf-8") as f:
                json.dump(optim_info, f, ensure_ascii=False, indent=4)

        return solution, op_status

    def backward(self, case: Case, solution: Solution, save_dir: Path = None):
        """
        Analyze tool usage for a single case.
        """
        # We need to extract tool usage history from the case
        tool_usage_history = self._get_tool_usage_history(case)

        # Construct prompt for backward analysis
        prompt_data = {
            "tool_usage_history": tool_usage_history,
            "toolkit_config": solution.agent_team.environment.shared_toolkit.to_dict()
        }

        # Use prompt formatter if meta_backward has template, otherwise simple format
        # Assuming meta_backward structure similar to other optimizers

        # For simplicity, if meta_backward is not fully configured, we might skip or use defaults.
        # But assuming it follows the pattern:
        if "order" in self.meta_backward:
            prompt = prompt_formatter.formulate_prompt(self.meta_backward, prompt_data)
        else:
             # Fallback or simple prompt
             prompt = f"Analyze the tool usage:\n{tool_usage_history}"

        _, content = self.llm.get_response(
            chat_messages=None,
            system_prompt="",
            last_prompt=prompt,
            stream=False
        )

        extracted_dict = OptimUtils.extract_data_from_response(
            content,
            self.meta_backward.get("extract_key", ["suggestion", "analyse"])
        )

        case.toolkit_suggestion = {
            "suggestion": extracted_dict.get("suggestion", ""),
            "analyse": extracted_dict.get("analyse", ""),
            "prompt": prompt,
            "response": content
        }

        if save_dir:
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            case.dump(save_dir / f"{case.case_id}.json")

    def formulate_optim_prompt(self, solution: Solution, case_list: List[Case]) -> str:
        # Collect suggestions from all cases
        suggestions = ""
        for i, case in enumerate(case_list):
            if hasattr(case, 'toolkit_suggestion'):
                suggestions += f"Case {i+1}:\nAnalyse: {case.toolkit_suggestion['analyse']}\nSuggestion: {case.toolkit_suggestion['suggestion']}\n\n"

        prompt_data = {
            "toolkit_config": json.dumps(solution.agent_team.environment.shared_toolkit.to_dict(), indent=2),
            "suggestions": suggestions
        }

        if "order" in self.meta_optim:
            prompt = prompt_formatter.formulate_prompt(self.meta_optim, prompt_data)
        else:
            prompt = f"Optimize the toolkit based on suggestions:\n{suggestions}"

        return prompt

    def apply_changes(self, solution: Solution, op_list: List[Dict[str, Any]]) -> bool:
        toolkit = solution.agent_team.environment.shared_toolkit
        changed = False

        for op in op_list:
            action = op.get("action")
            if action == "add_tool":
                tool_name = op.get("tool_name")
                tool_config = op.get("tool_config")
                if toolkit.add_tool(tool_name, tool_config):
                    changed = True
            elif action == "remove_tool":
                tool_name = op.get("tool_name")
                if toolkit.remove_tool(tool_name):
                    changed = True
            elif action == "update_tool_description":
                tool_name = op.get("tool_name")
                new_description = op.get("description")
                if toolkit.update_tool_description(tool_name, new_description):
                    changed = True

        return changed

    def _get_tool_usage_history(self, case: Case) -> str:
        # Extract relevant info from case.trajectory
        # Case.trajectory is a Trajectory object, containing a list of States.
        # Each State has an Action.
        # Action has tools_results_dict: Dict[str, dict]

        history = ""
        for state in case.trajectory.states:
             action = state.action
             # Assuming Action object is available on state.action
             # action.tools_results_dict seems to store tool results
             if action and hasattr(action, 'tools_results_dict') and action.tools_results_dict:
                 for tool_call_id, result in action.tools_results_dict.items():
                     # Construct a readable history
                     # We don't have exact tool input args here easily unless we parse content or look at how tools_results_dict is populated
                     # But at least we know a tool was called and the result.
                     history += f"Node: {state.node.node_name}, Role: {action.agent_role}\n"
                     history += f"Tool Call Result: {result}\n"

             # Also check content for context
             if action:
                 history += f"Agent Content: {action.content}\n"

        return history
