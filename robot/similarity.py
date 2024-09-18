"""
Author: Guo Fei
Date: 2024-09-14 15:27:02
LastEditors: Guo Fei
LastEditTime: 2024-09-14 15:27:03
Description:  借鉴 https://github.com/shibing624/similarities/blob/main/similarities/similarity.py#L15

Copyright (c) 2024 by SDARI, All Rights Reserved. 
"""

from typing import List, Union, Dict
from difflib import SequenceMatcher


def try_divide(x, y, val=0.0):
    """
    try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


def edit_distance(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        import Levenshtein

        d = Levenshtein.distance(str1, str2) / float(max(len(str1), len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1.0 - SequenceMatcher(lambda x: x == " ", str1, str2).ratio()
    return d


def longest_match_size(str1, str2):
    """最长公共子串长度"""
    sq = SequenceMatcher(None, str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def longest_match_ratio(str1, str2):
    """最长公共子串占比"""
    sq = SequenceMatcher(None, str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return try_divide(match.size, min(len(str1), len(str2)))


def num_of_common_sub_str(str1, str2):
    """
    求两个字符串的最长公共子串，同longest_match_size
    思想：建立一个二维数组，保存连续位相同与否的状态
    """
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    max_num = 0  # 最长匹配长度

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > max_num:
                    # 获取最大匹配长度
                    max_num = record[i + 1][j + 1]
    return max_num


class SimilarityABC:
    """
    Interface for similarity compute and search.

    In all instances, there is a corpus against which we want to perform the similarity search.
    For each similarity search, the input is a document or a corpus, and the output are the similarities
    to individual corpus documents.
    """

    def add_corpus(self, corpus: Union[List, Dict]):
        """
        Extend the corpus with new documents.
        corpus : list of str
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def similarity(self, a: Union[str, List], b: Union[str, List]):
        """
        Compute similarity between two texts.
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def distance(self, a: Union[str, List], b: Union[str, List]):
        """Compute cosine distance between two texts."""
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def most_similar(
        self, queries: Union[str, List, Dict], topn: int = 10
    ) -> List[List[Dict]]:
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: Dict[int(query_id), str(query_text)] or List[str] or str
        :param topn: int
        :return: List[List[Dict]], A list with one entry for each query. Each entry is a list of
            dict with the keys 'corpus_id', 'corpus_doc' and 'score', sorted by decreasing similarity scores.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def search(
        self, queries: Union[str, List, Dict], topn: int = 10
    ) -> List[List[Dict]]:
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: Dict[int(query_id), str(query_text)] or List[str] or str
        :param topn: int
        :return: List[List[Dict]], A list with one entry for each query. Each entry is a list of
            dict with the keys 'corpus_id', 'corpus_doc' and 'score', sorted by decreasing similarity scores.
        """
        return self.most_similar(queries, topn=topn)


class SameCharsSimilarity(SimilarityABC):
    """
    Compute text chars similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    不考虑文本字符位置顺序，基于相同字符数占比计算相似度
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None):
        super().__init__()
        self.corpus = {}

        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SameChars"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        if not corpus_new:
            return
        self.corpus.update(corpus_new)

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute Chars similarity between two texts.
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            if not sentence1 or not sentence2:
                return 0.0
            same = set(sentence1) & set(sentence2)
            similarity_score = max(
                len(same) / len(set(sentence1)), len(same) / len(set(sentence2))
            )
            return similarity_score

        return [
            calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)
        ]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(
        self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10
    ) -> List[List[Dict]]:
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, "__len__"):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = []
        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append(
                    {"corpus_id": corpus_id, "corpus_doc": doc, "score": score}
                )
            q_res = sorted(q_res, key=lambda x: x["score"], reverse=True)[:topn]
            result.append(q_res)
        return result


class SequenceMatcherSimilarity(SimilarityABC):
    """
    Compute text sequence matcher similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    考虑文本字符位置顺序，基于最长公共子串占比计算相似度
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None):
        super().__init__()
        self.corpus = {}

        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SequenceMatcher"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        if not corpus_new:
            return
        self.corpus.update(corpus_new)

    def similarity(
        self,
        a: Union[str, List[str]],
        b: Union[str, List[str]],
        min_same_len: int = 70,
        min_same_len_score: float = 0.9,
    ):
        """
        Compute Chars similarity between two texts.
        :param a:
        :param b:
        :param min_same_len:
        :param min_same_len_score:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            if not sentence1 or not sentence2:
                return 0.0
            same_size = longest_match_size(sentence1, sentence2)
            same_score = min_same_len_score if same_size > min_same_len else 0.0
            similarity_score = max(
                same_size / len(sentence1), same_size / len(sentence2), same_score
            )
            return similarity_score

        return [
            calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)
        ]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(
        self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10
    ) -> List[List[Dict]]:
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, "__len__"):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = []
        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append(
                    {"corpus_id": corpus_id, "corpus_doc": doc, "score": score}
                )
            q_res = sorted(q_res, key=lambda x: x["score"], reverse=True)[:topn]
            result.append(q_res)
        return result


if __name__ == "__main__":
    tool = SequenceMatcherSimilarity()
    tool.add_corpus(["首页", "操纵性", "快速性", "总结"])
    results = tool.most_similar("xxx", topn=3)
    print(results)
