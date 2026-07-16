from __future__ import annotations

import math
import re
from typing import Any

from lm_eval.api.metrics import is_degenerating_text


CHINESE_ANSWER_TYPE = {
    "Numerical": "数值",
    "Expression": "表达式",
    "Equation": "方程",
    "Interval": "区间",
}

ENGLISH_ANSWER_TYPE = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}


def _sympy_latex_tools():
    import sympy as sp
    from sympy import Eq, Pow, simplify, sympify
    from sympy.parsing.latex import parse_latex

    return sp, Eq, Pow, simplify, sympify, parse_latex


def answer_type_text(answer_type: str, is_chinese: bool, multiple_answer: bool) -> str:
    if ("Need_human_evaluate" in answer_type) or ("Tuple" in answer_type):
        return ""

    def single_type_text(single_answer_type: str) -> str:
        if "-" in single_answer_type:
            single_answer_type = single_answer_type[: single_answer_type.find("-")]
        mapping = CHINESE_ANSWER_TYPE if is_chinese else ENGLISH_ANSWER_TYPE
        for answer_type_key, answer_type_value in mapping.items():
            if answer_type_key in single_answer_type:
                return answer_type_value
        raise ValueError(f"Error parsing answer type {answer_type}.")

    if not multiple_answer:
        answer_text = single_type_text(answer_type)
        if is_chinese:
            return f"，答案类型为{answer_text}"
        return f"The answer of The problem should be {answer_text}. "

    if "," not in answer_type:
        answer_text = single_type_text(answer_type)
        if is_chinese:
            return f"，题目有多个答案，答案类型均为{answer_text}"
        return (
            f"The problem has multiple answers, each of them should be {answer_text}. "
        )

    answer_types = [single_type_text(item) for item in answer_type.split(",")]
    if len(set(answer_types)) == 1:
        answer_text = answer_types[0]
        if is_chinese:
            return f"，题目有多个答案，答案类型均为{answer_text}"
        return (
            f"The problem has multiple answers, each of them should be {answer_text}. "
        )

    if is_chinese:
        return f"题目有多个答案，答案类型分别为{'、'.join(answer_types)}"
    return (
        "The problem has multiple answers, with the answers in order being "
        f"{', '.join(answer_types)}. "
    )


def make_prompt(doc: dict[str, Any], subset: str) -> str:
    is_chinese = "_zh_" in subset
    is_math = "maths" in subset
    is_theorem_proving = "TP" in subset

    if is_chinese:
        subject = "数学" if is_math else "物理"
        if is_theorem_proving:
            return (
                f"以下是中国{subject}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。"
                "证明过程中使用的变量和公式请使用LaTeX格式表示。"
            )

        answer_text = answer_type_text(
            doc.get("answer_type", ""),
            is_chinese=True,
            multiple_answer=bool(doc.get("is_multiple_answer")),
        )
        boxed_text = (
            "\\boxed{用英文逗号连接的多个答案}"
            if doc.get("is_multiple_answer")
            else "\\boxed{答案}"
        )
        unit_text = ""
        if doc.get("unit"):
            boxed_text += "(单位)"
            unit_text = "，注意答案的单位不要放在\\boxed{}中"
        return (
            f"以下是中国{subject}竞赛中的解答题{answer_text}。请根据题目的要求和所提供的信息计算得出答案。"
            "解答过程和结果中使用的变量和公式请使用LaTeX格式表示。"
            f"请在最后以“所以最终答案是{boxed_text}。”显式给出结果{unit_text}。"
        )

    subject = "Math" if is_math else "Physics"
    if is_theorem_proving:
        return (
            f"The following is a theorem proving problem from an International {subject} competition. "
            "Please use logical reasoning and common theorems to prove the proposition in the problem "
            "according to the given requirements. Please use LaTeX format to represent the variables "
            "and formulas used in the proof."
        )

    boxed_text = (
        "\\boxed{multiple answers connected with commas}"
        if doc.get("is_multiple_answer")
        else "\\boxed{answer}"
    )
    unit_text = ""
    if doc.get("unit"):
        boxed_text += "(unit)"
        unit_text = (
            ", note that the unit of the answer should not be included in \\boxed{}"
        )
    answer_text = answer_type_text(
        doc.get("answer_type", ""),
        is_chinese=False,
        multiple_answer=bool(doc.get("is_multiple_answer")),
    )
    return (
        f"The following is an open-ended problem from an International {subject} competition. "
        f"{answer_text}Please calculate the answer according to the given requirements and the information provided. "
        "Please use LaTeX format to represent the variables and formulas used in the solution process and results. "
        f'Please end your solution with "So the final answer is {boxed_text}." and give the result explicitly{unit_text}.'
    )


def extract_answer(model_output: str, is_chinese: bool) -> str:
    patterns = (
        [
            r"所以最终答案是(.*)",
            r"最终答案(?:是|为)[:：]?\s*(.*)",
        ]
        if is_chinese
        else [
            r"So the final answer is (.*)",
            r"[Ff]inal [Aa]nswer:? (.*)",
            r"[Tt]he answer is:? (.*)",
        ]
    )
    for pattern in patterns:
        matches = re.findall(pattern, model_output)
        if matches:
            return matches[-1].strip()
    return model_output


def precision_from_error(error: Any) -> float | list[float]:
    if not error:
        return 1e-8
    if isinstance(error, str) and "," in error:
        return [float(item) if item else 1e-8 for item in error.split(",")]
    return float(error)


class OlympiadBenchScorer:
    def __init__(self):
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = None
        self.precision = 1e-8

    def judge(self, ground_truth, prediction, precision=1e-8) -> bool:
        precision = precision if isinstance(precision, list) else [precision]
        try:
            ground_truth, prediction = self.preprocess(ground_truth, prediction)
        except Exception:  # noqa: BLE001 - parser failures mean the answer is incorrect.
            return False
        if ground_truth == prediction:
            return True

        ground_truth = re.sub(r"[\u4e00-\u9fff]+", "", ground_truth)
        prediction = re.sub(r"[\u4e00-\u9fff]+", "", prediction)
        gt_items = self.trans_plus_minus_sign(self.split_by_comma(ground_truth))
        pred_items = self.trans_plus_minus_sign(self.split_by_comma(prediction))

        if len(precision) <= 1:
            precision = precision * len(gt_items)
        if len(gt_items) != len(pred_items):
            return False

        idx = -1
        while gt_items:
            idx = (idx + 1) % len(gt_items)
            gt_item = gt_items[idx]
            self.precision = precision[idx]
            for pred_item in pred_items:
                if self.is_equal(gt_item, pred_item):
                    gt_items.remove(gt_item)
                    pred_items.remove(pred_item)
                    precision.remove(self.precision)
                    break
            else:
                return False
        return True

    def split_by_comma(self, expr: str) -> list[str]:
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for index, char in enumerate(expr):
            if char in {"(", "["}:
                in_bracket_num += 1
            elif char in {")", "]"}:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:index].strip())
                start_idx = index + 1
        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())
        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list[str]) -> list[str]:
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)
        return new_expr_list

    def is_interval(self, expr: str) -> bool:
        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

    def sympy_sub_pi(self, expression_sympy):
        _, _, _, _, _, parse_latex = _sympy_latex_tools()
        if self.pi is None:
            self.pi = parse_latex("\\pi")
        return expression_sympy.subs(self.pi, math.pi)

    def is_equal(self, ground_truth: str, prediction: str) -> bool:
        if ground_truth == prediction and ground_truth and prediction:
            return True

        if self.is_interval(ground_truth) and self.is_interval(prediction):
            try:
                if self.interval_equal(ground_truth, prediction):
                    return True
            except Exception:  # noqa: BLE001 - invalid intervals are non-matches.
                return False

        try:
            if self.numerical_equal(ground_truth, prediction):
                return True
        except Exception:  # noqa: BLE001, S110 - try the next equivalence strategy.
            pass

        try:
            if self.expression_equal(ground_truth, prediction) and not (
                "=" in ground_truth and "=" in prediction
            ):
                return True
        except Exception:  # noqa: BLE001, S110 - try the next equivalence strategy.
            pass

        try:
            if self.equation_equal(ground_truth, prediction):
                return True
        except Exception:  # noqa: BLE001, S110 - no equivalence strategy matched.
            pass

        return False

    def numerical_equal(
        self, ground_truth: str, prediction: str, include_percentage: bool = True
    ) -> bool:
        reference = float(ground_truth)
        predicted = float(prediction)
        gt_result = (
            [reference / 100, reference, reference * 100]
            if include_percentage
            else [reference]
        )
        return any(abs(item - predicted) <= self.precision * 1.01 for item in gt_result)

    def expression_equal(self, exp1: str, exp2: str) -> bool:
        sp, _, _, simplify, sympify, parse_latex = _sympy_latex_tools()

        def extract_expression(expression: str) -> str:
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        expr1_sym = sympify(parse_latex(extract_expression(exp1)))
        expr2_sym = sympify(parse_latex(extract_expression(exp2)))

        if expr1_sym == expr2_sym:
            return True

        expr1_sym = self.sympy_sub_pi(expr1_sym)
        expr2_sym = self.sympy_sub_pi(expr2_sym)

        if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
            not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)
        ):
            return False
        if not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
            if not (
                self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)
            ):
                return False
            return abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01

        simplified_expr = simplify(expr1_sym - expr2_sym)
        return abs(simplified_expr.evalf()) < 1e-3

    def equation_equal(self, expression1: str, expression2: str) -> bool:
        _, Eq, _, simplify, _, parse_latex = _sympy_latex_tools()

        def simplify_equation(latex_eq: str):
            lhs, rhs = latex_eq.split("=")
            equation = Eq(parse_latex(lhs), parse_latex(rhs))
            return simplify(equation.lhs - equation.rhs)

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)
        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)
        return (
            division_result_1.is_Integer
            and division_result_1 != 0
            or division_result_2.is_Integer
            and division_result_2 != 0
        )

    def interval_equal(self, expression1: str, expression2: str) -> bool:
        def compare_two_interval(interval1: str, interval2: str) -> bool:
            if interval1[0] != interval2[0] or interval1[-1] != interval2[-1]:
                return False
            items_1 = interval1.strip("[]()").split(",")
            items_2 = interval2.strip("[]()").split(",")
            return all(
                self.expression_equal(item_1, item_2)
                for item_1, item_2 in zip(items_1, items_2, strict=True)
            )

        if expression1 == expression2:
            return True
        intervals_1 = expression1.split("\\cup")
        intervals_2 = expression2.split("\\cup")
        if len(intervals_1) != len(intervals_2):
            return False
        return all(
            compare_two_interval(interval1, interval2)
            for interval1, interval2 in zip(intervals_1, intervals_2, strict=True)
        )

    def preprocess(self, expression1: str, expression2: str) -> tuple[str, str]:
        return (
            self.special_symbol_replace(self.extract_boxed_content(str(expression1))),
            self.special_symbol_replace(self.extract_boxed_content(str(expression2))),
        )

    def extract_boxed_content(self, latex_str: str) -> str:
        boxed_matches = re.finditer(r"\\boxed{", latex_str)
        results = ""
        for match in boxed_matches:
            start_index = match.end()
            end_index = start_index
            stack = 1
            while stack > 0 and end_index < len(latex_str):
                if latex_str[end_index] == "{":
                    stack += 1
                elif latex_str[end_index] == "}":
                    stack -= 1
                end_index += 1
            if stack == 0:
                results += latex_str[start_index : end_index - 1] + ","
            else:
                raise ValueError("Mismatched braces in LaTeX string.")

        if results:
            return results

        last_line_ans = latex_str.strip().split("\n")[-1]
        answers = re.findall(r"\$(.*?)\$", last_line_ans)
        if answers:
            return ",".join(answers) + ","
        return latex_str

    def special_symbol_replace(self, expression: str) -> str:
        if "\\in " in expression:
            expression = expression.split("\\in ")[1]
        for signal, replacement in self.special_signal_map.items():
            expression = expression.replace(signal, replacement)
        expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")  # noqa: B005
        return re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", expression)

    def can_compute_power(self, expr) -> bool:
        _, _, Pow, _, _, _ = _sympy_latex_tools()
        if not isinstance(expr, Pow):
            return True
        base, exponent = expr.as_base_exp()
        if base.is_number and exponent.is_number:
            return abs(exponent.evalf()) <= 1000
        return False


_SCORER = None


def scorer() -> OlympiadBenchScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = OlympiadBenchScorer()
    return _SCORER


def process_results(
    doc: dict[str, Any], prediction: str, subset: str
) -> dict[str, int]:
    final_answer = doc.get("final_answer") or []
    gold = str(final_answer[0]) if final_answer else ""
    answer = extract_answer(prediction, is_chinese="_zh_" in subset)
    precision = precision_from_error(doc.get("error"))
    correct = scorer().judge(gold, answer, precision)
    return {
        "exact_match": int(correct),
        "degeneration": int(is_degenerating_text(prediction.lower())),
    }
