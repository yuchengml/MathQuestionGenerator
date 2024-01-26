from langchain_community.chat_models import ChatOpenAI
from prompt import generate_concept_prompt

if __name__ == '__main__':
    concept_prompt = generate_concept_prompt()
    chat = ChatOpenAI(temperature=0,
                      model="gpt-4-vision-preview",
                      max_tokens=1024)

    # Question
    q = "竹籃裡有24顆蘋果，紅蘋果有6顆，其他是青蘋果，青蘋果有幾顆？"

    # get a chat completion from the formatted messages
    msg = chat(concept_prompt.format_prompt(question=q).to_messages())
    print(msg.content)
