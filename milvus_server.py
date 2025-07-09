# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  10:27
# FileName: milvus_server
# Tools: PyCharm


import random
import logging
from milvus_src.milvus_connection import MilvusConnectionPool
from milvus_src.milvus_collection import CollectionManager
from milvus_src.milvus_entity import CollectionOperator
from milvus_src.milvus_db import DatabaseManager
from concurrent.futures import ThreadPoolExecutor
import asyncio
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusQA")

class MilvusService:
    def __init__(self, uri: str, token: str, max_workers: int = 100):
        # 连接池
        self.conn_pool = MilvusConnectionPool(uri, token, max_workers)
        # 线程池执行器
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # 初始化管理器
        self.db_manager = DatabaseManager(self.conn_pool)
        self.collection_manager = CollectionManager(self.conn_pool)
        # 复用Operator实例（避免频繁实例化）
        self.collection_operator = CollectionOperator(
            self.collection_manager,
            self.conn_pool,
            self.executor
        )

    def get_db_manager(self) -> DatabaseManager:
        return self.db_manager

    def get_collection_manager(self) -> CollectionManager:
        return self.collection_manager

    def get_collection_operator(self) -> CollectionOperator:
        return self.collection_operator

    async def close(self):
        await self.conn_pool.close_all()
        self.executor.shutdown()


async def demo_collection_operator():
    uri = "http://localhost:31006"
    token = "root:Milvus"

    # 创建线程池
    executor = ThreadPoolExecutor(max_workers=50)

    # 创建服务
    service = MilvusService(uri, token)
    coll_manager = service.get_collection_manager()
    operator = service.get_collection_operator()

    db_name = "qa_database_2"
    collection_name = "demo_qa_collection"

    try:
        # 创建并加载集合
        # await coll_manager.create_collection(
        #     collection_name=collection_name,
        #     db_name=db_name,
        #     vector_dim=768
        # )
        await coll_manager.load_collection(collection_name, db_name)


        # 准备数据
        questions = [
            "What is Python?",
            "How to install Python?",
            "What are Python decorators?"
        ]
        answers = [
            "Python is a popular programming language",
            "Download from python.org and run installer",
            "Decorators are functions that modify other functions"
        ]

        # 生成向量示例（真实应用中会用模型生成）
        vectors = [
            [random.random() for _ in range(768)],
            [random.random() for _ in range(768)],
            [random.random() for _ in range(768)]
        ]

        # # 插入数据
        data = [
            {"id": 4, "question_vector": vectors[0], "question_text": questions[0], "answer": answers[0]},
            # {"id": 5, "question_vector": vectors[1], "question_text": questions[1], "answer": answers[1]},
            # {"id": 6, "question_vector": vectors[2], "question_text": questions[2], "answer": answers[2]}
        ]

        # inserted_ids = await operator.insert(collection_name, data, db_name)
        # print(f"Inserted IDs: {inserted_ids}")


        #
        # 更新答案
        # update_data = [
        #     # {"id": 2, "answer": "Visit python.org/downloads and run the installer"}
        #     {"id": 5, "question_vector": vectors[1], "question_text": questions[1], "answer":"989876579868"}
        # ]
        # await operator.upsert(collection_name, update_data, db_name)
        # print("Answer updated")



        # 删除一条记录
        # await operator.delete(
        #     collection_name=collection_name,
        #     ids=[6],
        #     db_name=db_name
        # )
        # print("Deleted record with ID 6")

        # # 获取数据量
        counts = await operator.count(collection_name,db_name)
        print(f"Collection contains {counts} entities")

        # 查询特定数据
        # results = await operator.query(collection_name,"user_id == 12345",["id","question_text","answer"],db_name,[1,2,3])
        results = await operator.query(collection_name,"id == 5",["id","question_text","answer"],db_name)
        print(f"Query results：{results} ")

        # 搜索相似问题
        query_vector = [v * 1.05 for v in vectors[0]]  # 轻微修改的查询向量
        results = await operator.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=2,
            output_fields=["question_text", "answer"],
            db_name=db_name,
        )

        print(f'milvus 数据插入成功...........：{results}')
        from pprint import pprint
        pprint(results)
        '''
        输出内容格式：
        [
            {
                "question_text": "如何重置密码？",
                "answer": "访问设置页面点击重置链接",
                "similarity": 0.92
            },
            {
                "question_text": "忘记密码怎么办？",
                "answer": "通过邮箱接收重置链接",
                "similarity": 0.87
            }
        ]
        '''



    finally:
    #     # 清理资源
        await service.close()
        # pass
if __name__=="__main__":
    # 运行示例
    asyncio.run(demo_collection_operator())