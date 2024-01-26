from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

sys_prompt_msg = SystemMessagePromptTemplate.from_template(
    "You are a mathematics expert chatbot specializing in solving elementary school math problems. "
    "Possessing strong comprehension skills in Chinese mathematical concepts, you can articulate key "
    "logical points within math problems, covering areas such as mathematical concepts, reading comprehension, "
    "reasoning, and problems solving abilities. Your role is to construct a comprehensive problem-solving process "
    "based on these concepts."
)

concept_prompt_msg = HumanMessagePromptTemplate.from_template(
    "According to the following SAMPLE QUESTION, present several clear problem-solving concepts.\n"
    "SAMPLE QUESTION:\n"
    "{question}\n\n"
    "Your response in traditional chinese:\n"
)


def generate_concept_prompt():
    concept_prompt = ChatPromptTemplate(messages=[
        sys_prompt_msg,
        concept_prompt_msg
    ])
    return concept_prompt
