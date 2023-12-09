answer_evaluation_prompt_template = """
You are machine learning professor and you should choose better answer about given question.

Question: {question}

Answer 1: {answer_1}

Answer 2: {answer_2}

Which one is better for the question? No explanation Just choose one and print confidence which is bounded from 0.00 to 1.00 in two decimal places.

If there're no correct answer, print None for answer and confidence.


Better answer: 
Your confidence:
"""

explanation_evaluation_prompt_template = """
You are machine learning professor and you should choose better answer and logical reasoning about given question.
Remember the condition that the answer must be correct, and if the both of the answers are correct, choose more logical reasoning for the answer.

And if the answer is correct, you should measure the following metric for the corresponding logical reasoning.

1. Clarity: How clearly is the explanation presented? Can graduate students easily understand the content?
2. Relevance: Is the explanation relevant to the quiz questions and aligned with the expectations for graduate-level knowledge?
3. Clarification of Concepts: Does the explanation clarify required concepts, theories, or knowledge areas relevant to the quiz?
4. Conciseness: Is the explanation concise and focused, avoiding unnecessary details while maintaining completeness?
5. Professional Tone: Is the tone and style of the explanation appropriate for a graduate-level audience, reflecting a high level of professionalism?

Question: {question}

Answer 1: {answer_1}
Logical reasoning 1: {logical_reasoning_1}

Answer 2: {answer_2}
Logical reasoning 2: {logical_reasoning_2}

Which one is better for the question? No explanation Just choose one and print confidence which is bounded from 0.00 to 1.00 in two decimal places.
Remember you should only print confidence about logical reasoning when both of answer are correct, otherwise print None
If there're no correct answer, print None for answer and confidence.

In addtion, print scores of above five metrics for logical reasoning when the answer is correct. Each score for five metrics is bounded from 0.00 to 1.00 in two decimal places.
Remember if the answer is not correnct, you should print None for scores of above five metrics for logical reasoning



Choose Better answer:
Choose Better logical reasoning: 

Your confidence about answer you chose:
Your confidence about logical reasoning you chose:

Clarity for Logical reasoning 1:
Relevance for Logical reasoning 1:
Clarification of Concepts for Logical reasoning 1:
Conciseness for Logical reasoning 1:
Professional Tone for Logical reasoning 1:

Clarity for Logical reasoning 2:
Relevance for Logical reasoning 2:
Clarification of Concepts for Logical reasoning 2:
Conciseness for Logical reasoning 2:
Professional Tone for Logical reasoning 2:
"""

