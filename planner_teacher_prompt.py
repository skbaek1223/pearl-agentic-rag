from __future__ import annotations

import json
import re

MAX_CONTEXT_CHARS = 12_000


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _truncate_context(ctx_text: str, answer: str) -> str:
    sents = _SENT_SPLIT.split(ctx_text)

    if len(sents) >= 2:
        sup_idx = None
        for i, s in enumerate(sents):
            if answer in s:
                sup_idx = i
                break
        if sup_idx is None:
            sup_idx = len(sents) // 2

        sup_sent = sents[sup_idx]
        budget = MAX_CONTEXT_CHARS - len(sup_sent)

        before = " ".join(sents[:sup_idx])
        after = " ".join(sents[sup_idx + 1:])
        half = budget // 2

        if before and len(before) > half:
            before = "[...] " + before[-(half - 6):]
        if after and len(after) > half:
            after = after[:half - 6] + " [...]"

        parts = [p for p in (before, sup_sent, after) if p]
        return " ".join(parts)

    words = ctx_text.split()
    ans_idx = None
    for i, w in enumerate(words):
        if answer in w:
            ans_idx = i
            break
    if ans_idx is None:
        ans_idx = 0

    half = MAX_CONTEXT_CHARS // 2
    before_words = []
    used = 0
    for w in reversed(words[:ans_idx]):
        cost = len(w) + 1
        if used + cost > half:
            break
        before_words.append(w)
        used += cost
    before_words.reverse()

    after_words = []
    used = 0
    for w in words[ans_idx:]:
        cost = len(w) + 1
        if used + cost > half:
            break
        after_words.append(w)
        used += cost

    prefix = "[...] " if len(before_words) < ans_idx else ""
    suffix = " [...]" if len(after_words) < len(words) - ans_idx else ""
    return prefix + " ".join(before_words + after_words) + suffix


MODEL = "gpt-5"


SINGLEHOP_GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
    },
    "required": ["plan"],
    "additionalProperties": False,
}


MULTIHOP_GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 6,
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}


SINGLEHOP_SYSTEM = """You are an information retrieval planning expert. Given a single-hop question, a supporting context, and the answer, generate a single concrete retrieval step required to find the answer.

Output a JSON object with one field:
- "plan" (string): a single retrieval step.

Ensure:
1. The step should be targeted enough that a retrieval system would reliably return the supporting context and the answer can be confidently derived from it. Each step must target a specific answer-bearing attribute.
2. The plan must be derivable solely from the question. Do not include specific facts, names, or details that are only found in the answer or supporting context."""


SINGLEHOP_FEWSHOT = [
    {
        "input": {
            "question": "who sang the most wonderful summer of my life",
            "supporting_context": [
                {
                    "text": "Jackie Ward ( born Jacqueline McDonnell , 1941 ) , better known as Robin Ward , is an American singer , regarded as a `` one - hit wonder '' of 1963 million - selling song `` Wonderful Summer '' . However , using her real name she was highly accomplished and successful singing in groups ."
                },
            ],
            "answer": "Jackie Ward",
        },
        "output": {
            "plan": "Find the performer or singer of the song 'The Most Wonderful Summer of My Life'"
        },
    },
    {
        "input": {
            "question": "a spider with a skull on its back",
            "supporting_context": [
                {
                    "text": "Steatoda nobilis has a brown bulbous abdomen with cream coloured markings that are often likened to the shape of a skull . The legs are reddish - orange ."
                },
            ],
            "answer": "Steatoda nobilis",
        },
        "output": {
            "plan": "Identify the type of spider that has a skull-shaped marking on its back"
        },
    },
    {
        "input": {
            "question": "at what output is marginal product of labour the highest",
            "supporting_context": [
                {
                    "text": "When the MP is above the AP the AP will increase . Eventually the MP reaches it maximum value at the point of diminishing returns . Beyond this point MP will decrease ."
                },
            ],
            "answer": "at the point of diminishing returns",
        },
        "output": {
            "plan": "Determine the output level at which the marginal product of labour reaches its maximum value"
        },
    },
]


MULTIHOP_SYSTEM = """You are an information retrieval planning expert. Given a multi-hop question, context sources, and the answer, generate an ordered sequence of retrieval steps required to find the answer. Each step represents one concrete retrieval action.

Output a JSON object with one field:
- "steps" (array of strings): the ordered retrieval plan (2 to 6 steps).

Ensure:
1. The retrieval steps should be targeted enough that a retrieval system would reliably return all context sources needed to derive the answer. Each step must target a specific answer-bearing attribute.
2. The answer should be confidently derived from them.
3. Each step must be derivable solely from the question and the results of prior steps. Do not include specific facts, names, or details that are only found in the answer or supporting context.
4. Any specific descriptive detail already stated in the question itself -- a physical description, location modifier, era or date, named co-participant, or other qualifying phrase -- is public information, not an answer-only fact, and must be preserved in whichever step it belongs to (normally step 1, unless it genuinely cannot be used until a prior step's result is known). Do not silently drop it into a generic paraphrase, and do not defer it to a later "confirm/verify" step that does not actually need it. A downstream system executes step 1's exact wording as its first search query, so anything step 1 omits is lost, not just deprioritized.
5. Rule 4 is about PRESERVING details the question already gives you -- it is never license to STATE a name, date, title, or other specific identifier that the question does not give you, even if you can tell from the supporting context that it happens to be correct. If a step needs to name something not yet known (an entity only step 1's search result will reveal), phrase that step as a search for it ("Identify the actress who played X" / "Find the year Y was founded"), not as an assertion of it ("...played by Jane Doe" / "...founded in 1923"). A parenthetical aside stating a specific answer-shaped fact the question never mentioned is exactly the invented-fact failure rule 3 already forbids, just easier to miss when it's phrased as a clarifying aside instead of a bare claim -- watch for it in that form too."""


HOTPOTQA_FEWSHOT = [
    {
        "input": {
            "question": "Gunmen from Laredo starred which narrator of \"Frontier\"?",
            "supporting_context": [
                {
                    "text": "Gunmen from Laredo is a 1959 American western film produced and directed by Wallace MacDonald, which stars Robert Knapp, Maureen Hingert, and Walter Coy."
                },
                {
                    "text": "Walter Darwin Coy (January 31, 1909 – December 11, 1974) was an American stage, radio, film, and, principally, television actor, originally from Great Falls, Montana.  He was best known for narrating the NBC western anthology series, \"Frontier\", which aired early Sunday evenings in the 1955–1956 season."
                },
            ],
            "answer": "Walter Darwin Coy",
        },
        "output": {
            "steps": [
                "Find the actors who starred in 'Gunmen from Laredo'",
                "Determine which of these actors was the narrator of the series 'Frontier'",
            ],
        },
    },
    {
        "input": {
            "question": "Who invented the type of script used in autographs?",
            "supporting_context": [
                {
                    "text": "An autograph in Assyriology is the hand-copy of a cuneiform clay-tablet.  Producing an autograph is often the first step of a tablet's archaeological interpretation and the autograph is frequently the authoritative form that is published as source material."
                },
                {
                    "text": "Cuneiform script ( or or ), one of the earliest systems of writing, was invented by the Sumerians.  It is distinguished by its wedge-shaped marks on clay tablets, made by means of a blunt reed for a stylus."
                },
            ],
            "answer": "the Sumerians",
        },
        "output": {
            "steps": [
                "Determine the type of script used in autographs in the relevant field",
                "Find who invented that type of script",
            ],
        },
    },
    {
        "input": {
            "question": "The Bass Rock Lighthouse was next to what Castle?",
            "supporting_context": [
                {
                    "text": "Canty Bay is a coastal hamlet off the A198, in East Lothian, Scotland, situated opposite the Bass Rock and Tantallon Castle.  Settlements nearby include Auldhame, Scoughall, Seacliff, and the Peffer Sands."
                },
                {
                    "text": " The island belongs to Sir Hew Hamilton-Dalrymple, whose family acquired it in 1706, and before to the Lauder family for almost six centuries.  The Bass Rock Lighthouse was constructed on the rock in 1902, and the remains of an ancient chapel survive."
                },
            ],
            "answer": "Tantallon Castle",
        },
        "output": {
            "steps": [
                "Find the location of the Bass Rock Lighthouse",
                "Identify landmarks or castles located next to that location",
            ],
        },
    },
    {
        "input": {
            "question": "Which monument, often described as a granite obelisk over 50 feet tall, commemorates the founding of Springfield?",
            "supporting_context": [
                {
                    "text": "The Springfield Founders Memorial is a granite obelisk standing over 50 feet tall, erected in 1887 to commemorate the founding of Springfield by early settlers."
                },
            ],
            "answer": "Springfield Founders Memorial",
        },
        "output": {
            "steps": [
                "Identify the monument, described as a granite obelisk over 50 feet tall, that commemorates the founding of Springfield",
                "Confirm the monument's official name from a reliable source",
            ],
        },
    },
    {
        "input": {
            "question": "What position did Dr. Elena Vasquez hold while working at the central research campus of the Global Health Institute?",
            "supporting_context": [
                {
                    "text": "Dr. Elena Vasquez served as Chief Epidemiologist while working at the central research campus of the Global Health Institute from 2015 to 2020."
                },
            ],
            "answer": "Chief Epidemiologist",
        },
        "output": {
            "steps": [
                "Identify Dr. Elena Vasquez's position while working at the central research campus of the Global Health Institute",
                "Confirm the position title from a reliable biographical source",
            ],
        },
    },
]


def make_user_prompt(item: dict, dataset: str = "nq") -> str:
    lines = [f"Question: {item['question']}"]
    ctx_list = item.get("supporting_context", [])
    answer = item.get("answer", "")
    if ctx_list:
        if dataset == "nq":
            ctx_text = "\n\n".join(s["text"] for s in ctx_list)
            if len(ctx_text) > MAX_CONTEXT_CHARS:
                ctx_text = _truncate_context(ctx_text, answer)
            lines.append("Supporting context:\n" + ctx_text)
        else:
            per_source_budget = MAX_CONTEXT_CHARS // len(ctx_list)
            ctx_parts = []
            for i, s in enumerate(ctx_list, 1):
                text = s["text"]
                if len(text) > per_source_budget:
                    text = _truncate_context(text, answer)
                ctx_parts.append(f"[Source {i}] {text}")
            lines.append("Context sources:\n" + "\n\n".join(ctx_parts))
    lines.append(f"Answer: {answer}")
    return "\n\n".join(lines)


def build_fewshot_messages(fewshot: list[dict], dataset: str = "nq") -> list[dict]:
    messages = []
    for ex in fewshot:
        messages.append({"role": "user", "content": make_user_prompt(ex["input"], dataset)})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})
    return messages


TEACHER_CONFIG = {
    "nq": (SINGLEHOP_SYSTEM, SINGLEHOP_GOAL_SCHEMA, SINGLEHOP_FEWSHOT),
    "hotpotqa": (MULTIHOP_SYSTEM, MULTIHOP_GOAL_SCHEMA, HOTPOTQA_FEWSHOT),
}


def build_teacher_request(item: dict, dataset: str) -> dict:
    system_prompt, schema, fewshot = TEACHER_CONFIG[dataset]
    messages = [
        {"role": "system", "content": system_prompt},
        *build_fewshot_messages(fewshot, dataset),
        {"role": "user", "content": make_user_prompt(item, dataset)},
    ]
    return {
        "model": MODEL,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "goal_decomposition", "strict": True, "schema": schema},
        },
        "max_completion_tokens": 2048,
        "reasoning_effort": "minimal",
    }
