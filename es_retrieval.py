from elasticsearch import Elasticsearch, helpers
from typing import Optional, Dict
import json
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")


class ElasticObject:
    def __init__(self, host: str, port: Optional[str] = None) -> None:
        """_summary_

        Args:
            host (str): Host of an elasticsearch
            port (str): Port of an elasticsearch
        """

        self.host = host
        self.port = port

        if not self.host.startswith("http"):
            self.host = "http://" + self.host

        if self.port:
            self.host = self.host + ":" + self.port

        self._connect_server(self.host)

    def _connect_server(self, url: str):
        """_summary_

        Args:
            url (str): URL of an elasticsearch

        Returns:
            _type_: _description_
        """

        self.client = Elasticsearch(
            url, timeout=30, max_retries=10, retry_on_timeout=True
        )
        print(f"Connected to Elastic Server ({url})")

    def create_index(self, index_name: str, setting_path: str = "./settings.json"):
        """_summary_

        Args:
            index_name (str): Name of an index
            setting_path (str): Path of the setting file
        """

        with open(setting_path, encoding="utf-8") as f:
            settings = json.load(f)

        if self.client.indices.exists(index=index_name):
            print(f"{index_name} already exists.")
            usr_input = input("Do you want to delete? (Y/n)")
            if usr_input == "Y":
                self.client.indices.delete(index=index_name)

            else:
                return False

        self.client.indices.create(index=index_name, body=settings)
        print(f"Create an Index ({index_name})")
        return True

    def get_indices(self):
        indices = list(self.client.indices.get_alias().keys())
        return indices

    def delete_index(self, index_name: str):
        """_summary_

        Args:
            index_name (str): Name of the index
        """
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            print(f"Delete an Index ({index_name})")

        else:
            print(f"Not exist {index_name}")

    def insert_data(
        self,
        index_name: str,
        data_path: str = "../data/wikipedia_documents.json",
    ):
        """_summary_

        Args:
            index_name (str): Name of an index
            data_path (str): Path of the Document file
        """

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        print("Data Loding...")
        for i, v in enumerate(data.values()):
            doc = {
                "_index": index_name,
                "_type": "_doc",
                "_id": i,
                "text": v["text"],
                "title": v["title"],
            }

            docs.append(doc)

        helpers.bulk(self.client, docs)

        print("Data Upload Completed")
        self.document_count(index_name)

    def delete_data(self, index_name: str, doc_id):
        """_summary_

        Args:
            index_name (_type_): _description_
            doc_id (_type_): _description_
        """

        self.client.delete(index=index_name, id=doc_id)

        print(f"Deleted {doc_id} document.")
        
    def init_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            self.delete_index(index_name=index_name)
            
        self.create_index(index_name=index_name)
        print(f"Initialization...({index_name})")

    def document_count(self, index_name: str):

        counts = self.client.count(index=index_name, pretty=True)["count"]
        print(f"Number of documents to {index_name} is {counts}.")

    def search(self, index_name: str, question: str, topk: int = 10):

        body = {"query": {"bool": {"must": [{"match": {"text": question}}]}}}

        responses = self.client.search(index=index_name, body=body, size=topk)["hits"]["hits"]

        return responses


if __name__ == "__main__":

    es = ElasticObject("localhost:9200")
    es.create_index("wiki_docs")
    es.create_index("wiki_docs")
    es.delete_index("wiki_docs")
    es.create_index("wiki_docs")
    es.insert_data("wiki_docs")
    print(es.document_count("wiki_docs"))

    outputs = es.search("wiki_docs", "소백산맥의 동남부에 위치한 지역은?")

    for output in outputs:
        print("doc:", output["text"])
        print("score:", output["score"])
        print()
