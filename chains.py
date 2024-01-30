from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from prompt import get_concept_prompt, get_aug_questions_prompt
from structure import Process, AugmentedQuestions


def create_concept_chain(prompt_template: bool = True):
    parser = JsonOutputParser(pydantic_object=Process)
    concept_prompt = get_concept_prompt(output_format=parser.get_format_instructions())
    chat = ChatOpenAI(temperature=0,
                      model="gpt-4-vision-preview",
                      max_tokens=1024)

    if prompt_template:
        chain = concept_prompt | chat | parser
    else:
        chain = chat | parser
    return chain


def create_augment_chain():
    aug_parser = JsonOutputParser(pydantic_object=AugmentedQuestions)
    aug_questions_prompt = get_aug_questions_prompt(output_format=aug_parser.get_format_instructions())
    chat = ChatOpenAI(temperature=0,
                      model="gpt-4-vision-preview",
                      max_tokens=1024)

    aug_chain = aug_questions_prompt | chat | aug_parser
    return aug_chain


def create_concept_w_image_chain():
    parser = JsonOutputParser(pydantic_object=Process)
    concept_prompt = get_concept_prompt(output_format=parser.get_format_instructions())
    chat = ChatOpenAI(temperature=0,
                      model="gpt-4-vision-preview",
                      max_tokens=1024)

    chain = concept_prompt | chat | parser
    return chain
