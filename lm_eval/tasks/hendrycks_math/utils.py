"""Swiss six-shot MATH protocol using upstream's maintained math scorer.

Additional demonstrations originate from Swiss commit 4ac31da. The first four
retain upstream's corrected LaTeX and the shared scorer retains its normalization
and full-solution Math Verify fixes.
"""

from lm_eval.tasks.minerva_math.utils import (
    doc_to_text as doc_to_text,
    get_unnormalized_answer as get_unnormalized_answer,
    list_fewshot_samples as _minerva_examples,
    process_docs as process_docs,
    process_results as process_results,
)


def list_fewshot_samples() -> list[dict]:
    return _minerva_examples() + [
        {
            "problem": "One sphere is centered at $(3,-5,7)$ with radius $5 \\sqrt{5}.$ A second sphere "
            "is centered at $(0,1,1)$ with radius $2 \\sqrt{17}.$ The two spheres intersect "
            "in a circle. Find the radius of this circle.",
            "solution": "Let $A = (3,-5,7),$ the center of the first sphere, and let $B = (0,1,1),$ the "
            "center of the second sphere. We can compute that $AB = 9.$ Let $C$ be a point "
            "on the intersection of both spheres, so $AC = 5 \\sqrt{5}$ and $BC = 2 "
            "\\sqrt{17}.$ [asy] unitsize(0.3 cm);\n"
            "\n"
            "pair A, B, C;\n"
            "\n"
            "A = (0,0);\n"
            "B = (9,0);\n"
            "C = intersectionpoint(arc(A,5*sqrt(5),0,180),arc(B,2*sqrt(17),0,180));\n"
            "\n"
            "draw(A--B--C--cycle);\n"
            "draw(Circle(A,5*sqrt(5)));\n"
            "draw(Circle(B,2*sqrt(17)));\n"
            "\n"
            'label("$A$", A, W);\n'
            'label("$B$", B, S);\n'
            'label("$C$", C, N);\n'
            'label("$9$", (A + B)/2, S, red);\n'
            'label("$5 \\sqrt{5}$", (A + C)/2, NW, red, UnFill);\n'
            'label("$2 \\sqrt{17}$", (B + C)/2, E, red, UnFill);\n'
            "[/asy]\n"
            "\n"
            "By Heron's formula, we can compute that $[ABC] = 3 \\sqrt{149}.$\n"
            "\n"
            "Let $D$ be the foot of the perpendicular from $C$ to $\\overline{AB}.$\n"
            "\n"
            "[asy]\n"
            "unitsize(0.3 cm);\n"
            "\n"
            "pair A, B, C, D;\n"
            "\n"
            "A = (0,0);\n"
            "B = (9,0);\n"
            "C = intersectionpoint(arc(A,5*sqrt(5),0,180),arc(B,2*sqrt(17),0,180));\n"
            "D = (C.x,0);\n"
            "\n"
            "draw(A--B--C--cycle);\n"
            "draw(C--D);\n"
            "\n"
            'label("$A$", A, W);\n'
            'label("$B$", B, S);\n'
            'label("$C$", C, N);\n'
            'label("$D$", D, S);\n'
            "[/asy]\n"
            "\n"
            "Then the intersection of both spheres is the circle centered at $D$ with "
            "radius $CD.$ Thus,\n"
            "\\[CD = \\frac{2 [ABC]}{AB} = \\frac{6 \\sqrt{149}}{9} = \\boxed{\\frac{2 "
            "\\sqrt{149}}{3}}.\\] \n"
            "Final Answer: The final answer is $\\frac{2 \\sqrt{149}}{3}$.",
            "few_shot": "1",
        },
        {
            "problem": "Ryan has 3 red lava lamps and 3 blue lava lamps. He arranges them in a row on a "
            "shelf randomly, then turns 3 random lamps on. What is the probability that the "
            "leftmost lamp on the shelf is red, and the leftmost lamp which is turned on is "
            "also red?",
            "solution": "There are $\\binom{6}{3}=20$ ways for Ryan to arrange the lamps, and "
            "$\\binom{6}{3}=20$ ways for him to choose which lamps are on, giving "
            "$20\\cdot20=400$ total possible outcomes. There are two cases for the desired "
            "outcomes: either the left lamp is on, or it isn't. If the left lamp is on, "
            "there are $\\binom{5}{2}=10$ ways to choose which other lamps are on, and "
            "$\\binom{5}{2}=10$ ways to choose which other lamps are red. This gives "
            "$10\\cdot10=100$ possibilities. If the first lamp isn't on, there are "
            "$\\binom{5}{3}=10$ ways to choose which lamps are on, and since both the "
            "leftmost lamp and the leftmost lit lamp must be red, there are "
            "$\\binom{4}{1}=4$ ways to choose which other lamp is red. This case gives 40 "
            "valid possibilities, for a total of 140 valid arrangements out of 400. "
            "Therefore, the probability is $\\dfrac{140}{400}=\\boxed{\\dfrac{7}{20}}$. \n"
            "Final Answer: The final answer is $\\dfrac{7}{20}$.",
            "few_shot": "1",
        },
    ]
