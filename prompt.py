from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

sys_prompt_msg = SystemMessagePromptTemplate.from_template(
    "You are a mathematics expert chatbot specializing in solving elementary school math problems. "
    "Possessing strong comprehension skills in Chinese mathematical concepts, you can articulate key "
    "logical points within math problems, covering areas such as mathematical concepts, reading comprehension, "
    "reasoning, and problems solving abilities."
)

concept_prompt_msg = HumanMessagePromptTemplate.from_template(
    "According to the following SAMPLE QUESTION, list several clear problem-solving concepts.\n"
    "SAMPLE QUESTION:\n"
    "{question}\n\n"
    "The response must meet the understanding abilities of third grade elementary school students: {grade}\n"
    "Format the response like: {format_instructions}\n"
    "Your response in traditional chinese:\n"
)

concept_prompt_start_msg = HumanMessagePromptTemplate.from_template(
    "According to the following SAMPLE QUESTION and IMAGES, list several clear problem-solving concepts.\n"
    "SAMPLE QUESTION:\n"
    "{question}\n"
)

concept_prompt_end_msg = HumanMessagePromptTemplate.from_template(
    "The response must meet the understanding abilities of third grade elementary school students: {grade}\n"
    "Format the responselike: {format_instructions}\n"
    "Your response in traditional chinese:\n"
)

aug_questions_prompt_msg = HumanMessagePromptTemplate.from_template(
    "According sample_requests in the following CONCEPT, help students understand concepts and generate different "
    "questions that conform to the same problem-solving concepts, the recommended method is to design "
    "{n_questions} questions based on different situations, things, or numbers.\n"
    "CONCEPT:\n"
    "{concept}\n\n"
    "Format the response in traditional chinese like: {format_instructions}\n"
    "Your response:\n"
)


def get_concept_prompt(output_format: str):
    concept_prompt = ChatPromptTemplate(messages=[
        sys_prompt_msg,
        concept_prompt_msg
    ], partial_variables={"format_instructions": output_format})
    return concept_prompt


def get_aug_questions_prompt(output_format: str):
    aug_questions_prompt = ChatPromptTemplate(messages=[
        sys_prompt_msg,
        aug_questions_prompt_msg
    ], partial_variables={"format_instructions": output_format})
    return aug_questions_prompt
