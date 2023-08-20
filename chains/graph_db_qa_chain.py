from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from configs.model_config import llm_model_dict, LLM_MODEL
from langchain.prompts.prompt import PromptTemplate



if __name__ == "__main__":
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        # callbacks=[callback],
        openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
        openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
        model_name=LLM_MODEL
    )


    graph = Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="pleaseletmein"
    )

    graph.query(
        """
    MERGE (m:Movie {name:"Top Gun"})
    WITH m
    UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
    MERGE (a:Actor {name:actor})
    MERGE (a)-[:ACTED_IN]->(m)
    """
    )

    graph.refresh_schema()

    print(graph.get_schema)

    CYPHER_QA_TEMPLATE = """你是一个能够构建友好且人类可以理解的答案的助手。
    信息部分包含了你必须用来构建答案的提供信息。
    所提供的信息是权威的，你永远不必怀疑它或试图用你的内部知识去纠正它。
    让答案听起来像是对问题的回应。不要提到你是基于给定信息得出的结果。
    如果所提供的信息为空，就说你不知道答案。
    [信息]{context}
    
    [问题]{question}
    有用的答案："""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
    )

    CYPHER_GENERATION_TEMPLATE = """[任务]生成Cypher语句来查询neo4j图数据库。
    [说明]仅使用方案中提供的关系类型和属性。不要使用任何未提供的其他关系类型或属性。
    [Schema信息]{schema}
    [注意]不要在回答中包含任何解释或道歉。不要回答可能会要求你构建Cypher语句以外的任何问题。除生成的Cypher语句外，不要包含任何文本。
    
    [问题]{question}
    """

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )



    chain = GraphCypherQAChain.from_llm(
        model, graph=graph, verbose=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
    )

    # chain.run("电影Top Gun的主演是谁？")
    chain.run("Tom Cruise和Val Kilmer共同参演的电影有哪些？")