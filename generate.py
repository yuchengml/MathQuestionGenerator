import json

from langchain.schema.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from prompt import get_concept_prompt, get_aug_questions_prompt, concept_prompt_start_msg, concept_prompt_end_msg, \
    sys_prompt_msg
from structure import Process, AugmentedQuestions
from utils import encode_image


def generate(
        q: str = "竹籃裡有24顆蘋果，紅蘋果有6顆，其他是青蘋果，青蘋果有幾顆？",
        n_questions: int = 5
):
    parser = JsonOutputParser(pydantic_object=Process)
    aug_parser = JsonOutputParser(pydantic_object=AugmentedQuestions)

    concept_prompt = get_concept_prompt(output_format=parser.get_format_instructions())
    aug_questions_prompt = get_aug_questions_prompt(output_format=aug_parser.get_format_instructions())
    chat = ChatOpenAI(temperature=0,
                      model="gpt-4-vision-preview",
                      max_tokens=1024)

    chain = concept_prompt | chat | parser
    aug_chain = aug_questions_prompt | chat | aug_parser

    result = chain.invoke({"question": q})
    # print(result)

    for c in result['concepts']:
        q_results = aug_chain.invoke({"concept": c, "n_questions": n_questions})
        c['sample_questions'] += q_results['questions']

    result['question'] = q

    with open('result.json', 'w', ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(result)


def generate_w_images(
        q: str = "下圖中的虛線是對稱軸的話，請寫出編號。",
        n_questions: int = 5
):
    parser = JsonOutputParser(pydantic_object=Process)
    aug_parser = JsonOutputParser(pydantic_object=AugmentedQuestions)

    aug_questions_prompt = get_aug_questions_prompt(output_format=aug_parser.get_format_instructions())
    chat = ChatOpenAI(temperature=0,
                      model="gpt-4-vision-preview",
                      max_tokens=1024)

    # Path to your image
    image_path = "ex5-1.jpg"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    chain = chat | parser
    aug_chain = aug_questions_prompt | chat | aug_parser

    inputs = [
        sys_prompt_msg.format(),  # SystemMessage
        concept_prompt_start_msg.format(question=q),
        HumanMessage(
            content=[
                {"type": "text", "text": "IMAGES:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto",
                    },
                },
            ]
        ),
        concept_prompt_end_msg.format(format_instructions=parser.get_format_instructions())
    ]

    result = chain.invoke(inputs)
    # print(result)

    for c in result['concepts']:
        q_results = aug_chain.invoke({"concept": c, "n_questions": n_questions})
        c['sample_questions'] += q_results['questions']

    result['question'] = q

    with open('result.json', 'w', ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(result)


if __name__ == '__main__':
    # generate()
    generate_w_images()
