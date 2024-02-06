import ast
import os.path

import pandas as pd
from langchain.schema.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from chains import create_concept_chain, create_augment_chain
from prompt import get_concept_prompt, get_aug_questions_prompt, concept_prompt_start_msg, concept_prompt_end_msg, \
    sys_prompt_msg
from structure import Process, AugmentedQuestions
from utils import encode_image, dump_to_json


def generate(
        q: str = "竹籃裡有24顆蘋果，紅蘋果有6顆，其他是青蘋果，青蘋果有幾顆？",
        grade: str = "first",
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

    result = chain.invoke({"question": q, "grade": grade})
    # print(result)

    for c in result['concepts']:
        q_results = aug_chain.invoke({"concept": c, "n_questions": n_questions})
        c['sample_questions'] += q_results['questions']

    result['question'] = q

    dump_to_json(result)

    print(result)


def generate_w_images(
        q: str = "下圖中的虛線是對稱軸的話，請寫出編號。",
        grade: str = "fifth",
        n_questions: int = 5,
        image_path: str = "data/images/ex5-1.jpg"
):
    parser = JsonOutputParser(pydantic_object=Process)
    aug_parser = JsonOutputParser(pydantic_object=AugmentedQuestions)

    aug_questions_prompt = get_aug_questions_prompt(output_format=aug_parser.get_format_instructions())
    chat = ChatOpenAI(temperature=0,
                      model="gpt-4-vision-preview",
                      max_tokens=1024)

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
        concept_prompt_end_msg.format(format_instructions=parser.get_format_instructions(), grade=grade)
    ]

    result = chain.invoke(inputs)
    # print(result)

    for c in result['concepts']:
        q_results = aug_chain.invoke({"concept": c, "n_questions": n_questions})
        c['sample_questions'] += q_results['questions']

    result['question'] = q

    dump_to_json(result)

    print(result)


def generate_from_csv(
        csv_file: str = "data/question_list.csv",
        image_folder: str = "data/images",
        n_questions: int = 5,
):
    dtypes = {'unit': str, 'grade': str, 'question': str, 'hint': str, 'images': object}
    df = pd.read_csv(csv_file, dtype=dtypes)
    question_list = df['question'].to_list()
    grade_list = df['grade'].to_list()
    images_list = df['images'].apply(ast.literal_eval).to_list()

    chain_q = create_concept_chain()
    chain_img_q = create_concept_chain(prompt_template=False)
    aug_chain = create_augment_chain()
    parser = JsonOutputParser(pydantic_object=Process)

    results = []

    for q, image_files, grade in zip(question_list, images_list, grade_list):
        if q:
            if image_files:
                msg = HumanMessage(content=[{"type": "text", "text": "IMAGES:"}])
                for img_f in image_files:
                    # Getting the base64 string
                    base64_image = encode_image(os.path.join(image_folder, img_f))
                    img_data = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto",
                        },
                    }
                    msg.content.append(img_data)
                inputs = [
                    sys_prompt_msg.format(),  # SystemMessage
                    concept_prompt_start_msg.format(question=q),
                    msg,
                    concept_prompt_end_msg.format(format_instructions=parser.get_format_instructions(),
                                                  grade=grade)
                ]
                result = chain_img_q.invoke(inputs)
            else:
                result = chain_q.invoke({"question": q, "grade": grade})

            for c in result['concepts']:
                q_results = aug_chain.invoke({"concept": c, "n_questions": n_questions})
                c['sample_questions'] += q_results['questions']

            result = {'question': q, 'concepts': result['concepts']}
        else:
            result = {'question': q}

        results.append(result)

    dump_to_json({'results': results})


if __name__ == '__main__':
    generate()
    generate_w_images()
    generate_from_csv()
