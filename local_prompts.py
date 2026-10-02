import re

_MC_CHOICES_MARKER = re.compile(r"\n\n(?:\(A\)[ \t]|Answer Choices:)")


def strip_mc_choices(question: str) -> str:
    m = _MC_CHOICES_MARKER.search(question)
    return question[:m.start()].rstrip() if m else question


_QA_INSTRUCTION_COMMON = """You are a question answering assistant with the ability to perform searches and reason.

You must provide an accurate answer to the given question by performing searches to retrieve relevant information, and reasoning over the retrieved information to derive the answer.

To perform a search, write a search query in the format <|begin_search_query|> your query here <|end_search_query|>, written as concise keywords only.
Then, the system will then search for relevant content, and return helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.
After each search result, you will also reason about whether the evidence is sufficient, and decide the direction your reasoning should take next.

The final answer must be in the format \\boxed{YOUR_ANSWER}.
"""

SINGLE_HOP_EXAMPLE = """

Here is an example of a single-hop question that requires one retrieval step:

Question:
who holds the most women's wimbledon titles

Let's find the record holder for the most women's singles titles at Wimbledon.
<|begin_search_query|> the record holder for the most women's singles titles at Wimbledon. <|end_search_query|>
<|begin_search_result|> Martina Navratilova holds the record for the most women's singles titles at Wimbledon, having won the championship nine times between 1978 and 1990. <|end_search_result|>
[Reasoning Guide]: Sufficient.
You now have enough information to answer the original question "who holds the most women's wimbledon titles". Provide the final answer in the format \\boxed{YOUR_ANSWER}.

The search result clearly states Martina Navratilova holds the record with nine Wimbledon titles.
\\boxed{Martina Navratilova}
"""


MULTI_HOP_EXAMPLE = """

Here is an example of a multi-hop question that requires multiple retrieval steps:

Question:
The Oberoi family is part of a hotel company that has a head office in what city?

Let's identify the hotel company the Oberoi family is part of, then let's find the city where that hotel company's head office is located.
<|begin_search_query|> the hotel company the Oberoi family is part of. <|end_search_query|>
<|begin_search_result|> The Oberoi family is an Indian family involved in the hospitality industry. They are the founders and owners of The Oberoi Group, which operates luxury hotels and resorts. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above or the retrieval guide to inform your search strategy, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family is associated with The Oberoi Group. I still need to find the head office location, so I'll continue retrieval.
Let's find where The Oberoi Group's head office is located.
<|begin_search_query|> The Oberoi Group head office location <|end_search_query|>
<|begin_search_result|> The Oberoi Group is a hotel company founded in 1934. It operates luxury hotels, resorts, and cruisers in several countries. The company's head office is located in Delhi, India. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above or the retrieval guide to inform your search strategy, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family owns The Oberoi Group, whose head office is in Delhi.
\\boxed{Delhi}
"""


INSUFFICIENT_RETRY_EXAMPLE = """

Here is an example where the first retrieval is insufficient and a retry with a differently-angled approach is needed:

Question:
when did women get the vote in the uk

Let's find the year women gained the right to vote in the United Kingdom.
<|begin_search_query|> UK women vote year <|end_search_query|>
<|begin_search_result|> NONE <|end_search_result|>
[Reasoning Guide]: Insufficient.
Feel free to explore alternative paths, such as trying a different search query or taking other retrieval steps as needed, to derive an answer to the question: when did women get the vote in the uk.

The retrieval was entirely insufficient, so I should switch angles entirely. Instead of asking the date directly, search for the broader topic of women's suffrage history in the UK, which is more likely to appear in encyclopedic sources.
<|begin_search_query|> women suffrage UK history <|end_search_query|>
"""


_SINGLE_HOP_FINAL_ANSWER_NOTE = """
If more than one entity could serve as the answer, name only the one that most directly and specifically answers what was asked.
"""


def get_single_qa_instruction(include_final_answer_note: bool = True):
    note = _SINGLE_HOP_FINAL_ANSWER_NOTE if include_final_answer_note else ""
    return _QA_INSTRUCTION_COMMON + note + SINGLE_HOP_EXAMPLE + INSUFFICIENT_RETRY_EXAMPLE


def get_multi_qa_instruction():
    return _QA_INSTRUCTION_COMMON + MULTI_HOP_EXAMPLE + INSUFFICIENT_RETRY_EXAMPLE


SINGLE_HOP_FORCED_SEARCH_EXAMPLE = """

Here is an example of a single-hop question that requires one retrieval step:

Question:
who holds the most women's wimbledon titles

<|begin_search_query|> who holds the most women's wimbledon titles <|end_search_query|>
<|begin_search_result|> Martina Navratilova holds the record for the most women's singles titles at Wimbledon, having won the championship nine times between 1978 and 1990. <|end_search_result|>
[Reasoning Guide]: Sufficient.
You now have enough information to answer the original question "who holds the most women's wimbledon titles". Provide the final answer in the format \\boxed{YOUR_ANSWER}.

The search result clearly states Martina Navratilova holds the record with nine Wimbledon titles.
\\boxed{Martina Navratilova}
"""


MULTI_HOP_FORCED_SEARCH_EXAMPLE = """

Here is an example of a multi-hop question that requires multiple retrieval steps:

Question:
The Oberoi family is part of a hotel company that has a head office in what city?

<|begin_search_query|> The Oberoi family is part of a hotel company that has a head office in what city? <|end_search_query|>
<|begin_search_result|> The Oberoi family is an Indian family involved in the hospitality industry. They are the founders and owners of The Oberoi Group, which operates luxury hotels and resorts. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above to inform your search strategy, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family is associated with The Oberoi Group. I still need to find the head office location, so I'll continue retrieval.
Let's find where The Oberoi Group's head office is located.
<|begin_search_query|> The Oberoi Group head office location <|end_search_query|>
<|begin_search_result|> The Oberoi Group is a hotel company founded in 1934. It operates luxury hotels, resorts, and cruisers in several countries. The company's head office is located in Delhi, India. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above to inform your search strategy, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family owns The Oberoi Group, whose head office is in Delhi.
\\boxed{Delhi}
"""


INSUFFICIENT_RETRY_FORCED_SEARCH_EXAMPLE = """

Here is an example where the first retrieval is insufficient and a retry with a differently-angled approach is needed:

Question:
when did women get the vote in the uk

<|begin_search_query|> when did women get the vote in the uk <|end_search_query|>
<|begin_search_result|> NONE <|end_search_result|>
[Reasoning Guide]: Insufficient.
Feel free to explore alternative paths, such as trying a different search query or taking other retrieval steps as needed, to derive an answer to the question: when did women get the vote in the uk.

The retrieval was entirely insufficient, so I should switch angles entirely. Instead of asking the date directly, search for the broader topic of women's suffrage history in the UK, which is more likely to appear in encyclopedic sources.
<|begin_search_query|> women suffrage UK history <|end_search_query|>
"""


def get_single_qa_forced_search_instruction(include_final_answer_note: bool = True):
    note = _SINGLE_HOP_FINAL_ANSWER_NOTE if include_final_answer_note else ""
    return _QA_INSTRUCTION_COMMON + note + SINGLE_HOP_FORCED_SEARCH_EXAMPLE + INSUFFICIENT_RETRY_FORCED_SEARCH_EXAMPLE


def get_multi_qa_forced_search_instruction():
    return _QA_INSTRUCTION_COMMON + MULTI_HOP_FORCED_SEARCH_EXAMPLE + INSUFFICIENT_RETRY_FORCED_SEARCH_EXAMPLE


_SINGLE_HOP_NO_PLANNER_EXAMPLE = """

Here is an example of a single-hop question that requires one retrieval step:

Question:
who holds the most women's wimbledon titles

I need to find the record holder for the most women's singles titles at Wimbledon.
<|begin_search_query|> the record holder for the most women's singles titles at Wimbledon. <|end_search_query|>
<|begin_search_result|> Martina Navratilova holds the record for the most women's singles titles at Wimbledon, having won the championship nine times between 1978 and 1990. <|end_search_result|>
[Reasoning Guide]: Sufficient.
You now have enough information to answer the original question "who holds the most women's wimbledon titles". Provide the final answer in the format \\boxed{YOUR_ANSWER}.

The search result clearly states Martina Navratilova holds the record with nine Wimbledon titles.
\\boxed{Martina Navratilova}
"""


_MULTI_HOP_NO_PLANNER_EXAMPLE = """

Here is an example of a multi-hop question that requires multiple retrieval steps:

Question:
The Oberoi family is part of a hotel company that has a head office in what city?

I need to first identify the hotel company the Oberoi family is part of.
<|begin_search_query|> the hotel company the Oberoi family is part of. <|end_search_query|>
<|begin_search_result|> The Oberoi family is an Indian family involved in the hospitality industry. They are the founders and owners of The Oberoi Group, which operates luxury hotels and resorts. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family is associated with The Oberoi Group. I still need to find the head office location, so I'll continue retrieval.
<|begin_search_query|> The Oberoi Group head office location <|end_search_query|>
<|begin_search_result|> The Oberoi Group is a hotel company founded in 1934. It operates luxury hotels, resorts, and cruisers in several countries. The company's head office is located in Delhi, India. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family owns The Oberoi Group, whose head office is in Delhi.
\\boxed{Delhi}
"""


_INSUFFICIENT_RETRY_NO_PLANNER_EXAMPLE = """

Here is an example where the first retrieval is insufficient and a retry with a differently-angled approach is needed:

Question:
when did women get the vote in the uk

I need to find the year women gained the right to vote in the United Kingdom.
<|begin_search_query|> UK women vote year <|end_search_query|>
<|begin_search_result|> NONE <|end_search_result|>
[Reasoning Guide]: Insufficient.
Feel free to explore alternative paths, such as trying a different search query or taking other retrieval steps as needed, to derive an answer to the question: when did women get the vote in the uk.

The retrieval was entirely insufficient, so I should switch angles entirely. Instead of asking the date directly, search for the broader topic of women's suffrage history in the UK, which is more likely to appear in encyclopedic sources.
<|begin_search_query|> women suffrage UK history <|end_search_query|>
"""


_SINGLE_HOP_NO_EVALUATOR_EXAMPLE = """

Here is an example of a single-hop question that requires one retrieval step:

Question:
who holds the most women's wimbledon titles

Let's find the record holder for the most women's singles titles at Wimbledon.
<|begin_search_query|> the record holder for the most women's singles titles at Wimbledon. <|end_search_query|>
<|begin_search_result|> Martina Navratilova holds the record for the most women's singles titles at Wimbledon, having won the championship nine times between 1978 and 1990. <|end_search_result|>
The search result clearly states Martina Navratilova holds the record with nine Wimbledon titles. I now have enough information to answer.
\\boxed{Martina Navratilova}
"""


_MULTI_HOP_NO_EVALUATOR_EXAMPLE = """

Here is an example of a multi-hop question that requires multiple retrieval steps:

Question:
The Oberoi family is part of a hotel company that has a head office in what city?

Let's identify the hotel company the Oberoi family is part of, then let's find the city where that hotel company's head office is located.
<|begin_search_query|> the hotel company the Oberoi family is part of. <|end_search_query|>
<|begin_search_result|> The Oberoi family is an Indian family involved in the hospitality industry. They are the founders and owners of The Oberoi Group, which operates luxury hotels and resorts. <|end_search_result|>
The Oberoi family is associated with The Oberoi Group. I still need to find the head office location, so I'll continue retrieval.
Let's find where The Oberoi Group's head office is located.
<|begin_search_query|> The Oberoi Group head office location <|end_search_query|>
<|begin_search_result|> The Oberoi Group is a hotel company founded in 1934. It operates luxury hotels, resorts, and cruisers in several countries. The company's head office is located in Delhi, India. <|end_search_result|>
The Oberoi family owns The Oberoi Group, whose head office is in Delhi. I now have enough information to answer.
\\boxed{Delhi}
"""


_INSUFFICIENT_RETRY_NO_EVALUATOR_EXAMPLE = """

Here is an example where the first retrieval is insufficient and a retry with a differently-angled approach is needed:

Question:
when did women get the vote in the uk

Let's find the year women gained the right to vote in the United Kingdom.
<|begin_search_query|> UK women vote year <|end_search_query|>
<|begin_search_result|> NONE <|end_search_result|>
That retrieval was entirely insufficient, so I should switch angles entirely. Instead of asking the date directly, search for the broader topic of women's suffrage history in the UK, which is more likely to appear in encyclopedic sources.
<|begin_search_query|> women suffrage UK history <|end_search_query|>
"""


def get_single_qa_no_evaluator_instruction(include_final_answer_note: bool = True):
    note = _SINGLE_HOP_FINAL_ANSWER_NOTE if include_final_answer_note else ""
    return (_QA_INSTRUCTION_COMMON + note + _SINGLE_HOP_NO_EVALUATOR_EXAMPLE
            + _INSUFFICIENT_RETRY_NO_EVALUATOR_EXAMPLE)


def get_multi_qa_no_evaluator_instruction():
    return (_QA_INSTRUCTION_COMMON + _MULTI_HOP_NO_EVALUATOR_EXAMPLE
            + _INSUFFICIENT_RETRY_NO_EVALUATOR_EXAMPLE)


_SINGLE_HOP_NO_PLANNER_NO_EVALUATOR_EXAMPLE = """

Here is an example of a single-hop question that requires one retrieval step:

Question:
who holds the most women's wimbledon titles

I need to find the record holder for the most women's singles titles at Wimbledon.
<|begin_search_query|> the record holder for the most women's singles titles at Wimbledon. <|end_search_query|>
<|begin_search_result|> Martina Navratilova holds the record for the most women's singles titles at Wimbledon, having won the championship nine times between 1978 and 1990. <|end_search_result|>
The search result clearly states Martina Navratilova holds the record with nine Wimbledon titles. I now have enough information to answer.
\\boxed{Martina Navratilova}
"""


_MULTI_HOP_NO_PLANNER_NO_EVALUATOR_EXAMPLE = """

Here is an example of a multi-hop question that requires multiple retrieval steps:

Question:
The Oberoi family is part of a hotel company that has a head office in what city?

I need to first identify the hotel company the Oberoi family is part of.
<|begin_search_query|> the hotel company the Oberoi family is part of. <|end_search_query|>
<|begin_search_result|> The Oberoi family is an Indian family involved in the hospitality industry. They are the founders and owners of The Oberoi Group, which operates luxury hotels and resorts. <|end_search_result|>
The Oberoi family is associated with The Oberoi Group. I still need to find the head office location, so I'll continue retrieval.
<|begin_search_query|> The Oberoi Group head office location <|end_search_query|>
<|begin_search_result|> The Oberoi Group is a hotel company founded in 1934. It operates luxury hotels, resorts, and cruisers in several countries. The company's head office is located in Delhi, India. <|end_search_result|>
The Oberoi family owns The Oberoi Group, whose head office is in Delhi. I now have enough information to answer.
\\boxed{Delhi}
"""


_INSUFFICIENT_RETRY_NO_PLANNER_NO_EVALUATOR_EXAMPLE = """

Here is an example where the first retrieval is insufficient and a retry with a differently-angled approach is needed:

Question:
when did women get the vote in the uk

I need to find the year women gained the right to vote in the United Kingdom.
<|begin_search_query|> UK women vote year <|end_search_query|>
<|begin_search_result|> NONE <|end_search_result|>
That retrieval was entirely insufficient, so I should switch angles entirely. Instead of asking the date directly, search for the broader topic of women's suffrage history in the UK, which is more likely to appear in encyclopedic sources.
<|begin_search_query|> women suffrage UK history <|end_search_query|>
"""


def get_single_qa_no_planner_no_evaluator_instruction(include_final_answer_note: bool = True):
    note = _SINGLE_HOP_FINAL_ANSWER_NOTE if include_final_answer_note else ""
    return (_QA_INSTRUCTION_COMMON + note + _SINGLE_HOP_NO_PLANNER_NO_EVALUATOR_EXAMPLE
            + _INSUFFICIENT_RETRY_NO_PLANNER_NO_EVALUATOR_EXAMPLE)


def get_simpledeepsearcher_qa_instruction(max_search_limit: int = 10) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"Whenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, please perform a search to gather more accurate, up-to-date, or specific information. You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {max_search_limit}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- Do not generate <|begin_search_result|> and <|end_search_result|> tags yourself.\n\n"
    )


def get_simpledeepsearcher_task_instruction_openqa(question: str) -> str:
    return (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{question}\n\n'
    )


def get_simpledeepsearcher_extraction_instruction(prev_reasoning: str, search_query: str, document: str) -> str:
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- Present the helpful information for current search query: beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

**Inputs:**
- **Previous Reasoning Steps:**
{prev_reasoning}

- **Current Search Query:**
{search_query}

- **Searched Web Pages:**
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_multi_qa_no_planner_no_evaluator_instruction():
    return (_QA_INSTRUCTION_COMMON + _MULTI_HOP_NO_PLANNER_NO_EVALUATOR_EXAMPLE
            + _INSUFFICIENT_RETRY_NO_PLANNER_NO_EVALUATOR_EXAMPLE)


_QA_ZERO_SHOT_INSTRUCTION = """You are a question answering assistant. Answer the given question directly using only your own knowledge and reasoning -- no external search tool is available for this question.

The final answer must be in the format \\boxed{YOUR_ANSWER}.
"""


def get_qa_zero_shot_instruction(include_final_answer_note: bool = True):
    note = _SINGLE_HOP_FINAL_ANSWER_NOTE if include_final_answer_note else ""
    return _QA_ZERO_SHOT_INSTRUCTION + note


def get_single_qa_no_planner_instruction(include_final_answer_note: bool = True):
    note = _SINGLE_HOP_FINAL_ANSWER_NOTE if include_final_answer_note else ""
    return (_QA_INSTRUCTION_COMMON + note + _SINGLE_HOP_NO_PLANNER_EXAMPLE
            + _INSUFFICIENT_RETRY_NO_PLANNER_EXAMPLE)


def get_multi_qa_no_planner_instruction():
    return (_QA_INSTRUCTION_COMMON + _MULTI_HOP_NO_PLANNER_EXAMPLE
            + _INSUFFICIENT_RETRY_NO_PLANNER_EXAMPLE)


def get_extractor_instruction(question, recent_reasoning, search_query, documents, is_multi_hop=False, dataset_name=None):
    question = strip_mc_choices(question)
    if is_multi_hop:
        candidate_rule = "- If several candidates could each plausibly answer the search query, pick the one that most directly and specifically answers it as asked, not a more recent or more prominent one."
        hop_rules = "- If no relevant fact is found, output NONE."
    else:
        candidate_rule = """- If several candidates could each plausibly answer the search query: pick the one that most directly and specifically answers it as asked IF the search query itself already gives enough context to tell them apart; otherwise extract all of them together with whatever distinguishing detail (date, context, series, etc.) the document gives, instead of forcing a pick or collapsing them into one tidy sentence."""
        hop_rules = """- If the ENTIRE answer is a single word, number, or short phrase AND there is exactly one relevant fact (nothing else relevant found), restate it as one complete sentence that directly answers the Current Search Query (e.g. "12" -> "12 avenues radiate from the Arc de Triomphe."), instead of outputting the bare word/phrase alone.
- If no relevant fact is found, output NONE."""
    extra_rules = "\n".join(r for r in (candidate_rule, hop_rules) if r)
    return f"""You are an extraction assistant. Read the retrieved documents and extract the facts relevant to the Current Search Query below -- judge relevance against the Current Search Query only, not against whether the information fully answers the original multi-hop Question (a document that resolves only the current hop is still relevant, even if it doesn't finish the Question).

Rules:
- Use only the retrieved documents; do not infer, speculate, or use outside knowledge.
{extra_rules}

Answer directly and concisely. Do not restate these rules or deliberate about them at length.

Output format:
**Extracted Information**

[the extracted fact, or NONE]

Question: {question}
Current Search Query: {search_query}
Recent Reasoning: {recent_reasoning}
Retrieved Documents: {documents}

Now you should read the retrieved documents and find helpful information based on the current search query "{search_query}" and recent reasoning.
"""
