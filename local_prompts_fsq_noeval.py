from local_prompts import _QA_INSTRUCTION_COMMON, _SINGLE_HOP_FINAL_ANSWER_NOTE


SINGLE_HOP_FORCED_SEARCH_NO_EVALUATOR_EXAMPLE = """

Here is an example of a single-hop question that requires one retrieval step:

Question:
who holds the most women's wimbledon titles

<|begin_search_query|> who holds the most women's wimbledon titles <|end_search_query|>
<|begin_search_result|> Martina Navratilova holds the record for the most women's singles titles at Wimbledon, having won the championship nine times between 1978 and 1990. <|end_search_result|>
The search result clearly states Martina Navratilova holds the record with nine Wimbledon titles. I now have enough information to answer.
\\boxed{Martina Navratilova}
"""


MULTI_HOP_FORCED_SEARCH_NO_EVALUATOR_EXAMPLE = """

Here is an example of a multi-hop question that requires multiple retrieval steps:

Question:
The Oberoi family is part of a hotel company that has a head office in what city?

<|begin_search_query|> The Oberoi family is part of a hotel company that has a head office in what city? <|end_search_query|>
<|begin_search_result|> The Oberoi family is an Indian family involved in the hospitality industry. They are the founders and owners of The Oberoi Group, which operates luxury hotels and resorts. <|end_search_result|>
The Oberoi family is associated with The Oberoi Group. I still need to find the head office location, so I'll continue retrieval.
Let's find where The Oberoi Group's head office is located.
<|begin_search_query|> The Oberoi Group head office location <|end_search_query|>
<|begin_search_result|> The Oberoi Group is a hotel company founded in 1934. It operates luxury hotels, resorts, and cruisers in several countries. The company's head office is located in Delhi, India. <|end_search_result|>
The Oberoi family owns The Oberoi Group, whose head office is in Delhi. I now have enough information to answer.
\\boxed{Delhi}
"""


INSUFFICIENT_RETRY_FORCED_SEARCH_NO_EVALUATOR_EXAMPLE = """

Here is an example where the first retrieval is insufficient and a retry with a differently-angled approach is needed:

Question:
when did women get the vote in the uk

<|begin_search_query|> when did women get the vote in the uk <|end_search_query|>
<|begin_search_result|> NONE <|end_search_result|>
That retrieval was entirely insufficient, so I should switch angles entirely. Instead of asking the date directly, search for the broader topic of women's suffrage history in the UK, which is more likely to appear in encyclopedic sources.
<|begin_search_query|> women suffrage UK history <|end_search_query|>
"""


def get_single_qa_forced_search_no_evaluator_instruction(include_final_answer_note: bool = True):
    note = _SINGLE_HOP_FINAL_ANSWER_NOTE if include_final_answer_note else ""
    return (_QA_INSTRUCTION_COMMON + note + SINGLE_HOP_FORCED_SEARCH_NO_EVALUATOR_EXAMPLE
            + INSUFFICIENT_RETRY_FORCED_SEARCH_NO_EVALUATOR_EXAMPLE)


def get_multi_qa_forced_search_no_evaluator_instruction():
    return (_QA_INSTRUCTION_COMMON + MULTI_HOP_FORCED_SEARCH_NO_EVALUATOR_EXAMPLE
            + INSUFFICIENT_RETRY_FORCED_SEARCH_NO_EVALUATOR_EXAMPLE)
