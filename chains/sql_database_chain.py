from langchain.chat_models import ChatOpenAI
from configs.model_config import llm_model_dict, LLM_MODEL, SQLALCHEMY_DATABASE_URI
from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

model = ChatOpenAI(
    streaming=True,
    verbose=True,
    # callbacks=[callback],
    openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
    openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
    model_name=LLM_MODEL
)


db = SQLDatabase.from_uri(
    SQLALCHEMY_DATABASE_URI,
    include_tables=['knowledge_base'],  # we include only one table to save tokens in the prompt :)
    sample_rows_in_table_info=2
)


human_prompt = "{input}"
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [("human", "我们来玩成语接龙，我先来，生龙活虎"),
     ("ai", "虎头虎脑"),
     ("human", "{input}")])

chain = create_sql_query_chain(llm=model, db=db)
response = chain.invoke({"question": "How many employees are there"})
print(response)