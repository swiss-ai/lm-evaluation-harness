# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import sqlite3
import numpy as np
import pickle as pkl
from typing import List
import concurrent.futures
import threading

from lm_eval.tasks.hallulens.cache import Cache
from lm_eval.tasks.hallulens.retrieval import DocDB
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


class LongWikiDB(DocDB):
    def __init__(self, db_path: str, data_path: str = None):
        # import DocDB from FactScore
        super(LongWikiDB, self).__init__(db_path, data_path)
        self.title_db_path = db_path.replace(".db", "-title.db")
        self._local = threading.local()
        self.SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

    def _get_title_connection(self):
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.title_db_path)
        return self._local.connection

    def get_relevant_titles(self, entity: str):
        conn = self._get_title_connection()
        cursor = conn.cursor()
        entity = entity.replace("'", "''")
        cursor.execute(
            "SELECT title_name FROM titles WHERE title_name LIKE ?",
            ("%" + entity + "%",),
        )
        results = cursor.fetchall()
        cursor.close()
        results = [r[0] for r in results]
        return results

    def get_whole_passages(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT title, text FROM documents")
        results = cursor.fetchall()
        results = [r for r in results]
        results = [
            {"title": r[0], "text": para}
            for r in results
            for para in r[1].split(self.SPECIAL_SEPARATOR)
        ]
        return results


class LongWikiRetrieval(object):
    def __init__(
        self,
        db,
        cache_base_path,
        embed_cache_path,
        retrieval_type="gtr-t5-large",
        batch_size=None,
        debugging=False,
    ):
        self.db = db
        self.CACHE_BASE_PATH = cache_base_path
        self.embed_cache_path = embed_cache_path
        self._embed_lock = threading.Lock()
        self._encode_lock = threading.Lock()  # serialize GPU encoding calls
        self.load_cache()

        self.retrieval_type = retrieval_type
        self.batch_size = batch_size

        ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        ner_model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-large-NER"
        )
        self.ner = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            batch_size=64,
            device=0,
        )

        self.encoder = None
        self._query_vec_cache = {}
        self.not_existing_pages = set()
        self.debugging = debugging

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            "sentence-transformers/" + self.retrieval_type, device=0
        )
        self.encoder = encoder
        assert self.batch_size is not None

    def load_cache(self):
        # cache
        self.relevant_pages_cache_path = f"{self.CACHE_BASE_PATH}/relevant_pages_cache.json"  # key: entity, value: list of relevant page titles
        self.Q_NER_cache_path = f"{self.CACHE_BASE_PATH}/question_to_ner_cache.json"
        self.cache_path = f"{self.CACHE_BASE_PATH}/cache.json"
        self.embed_cache_path = self.embed_cache_path

        self.add_n = 0
        self.add_n_embed = 0

        # embedding cache
        if os.path.exists(self.embed_cache_path):
            try:
                with open(self.embed_cache_path, "rb") as f:
                    self.embed_cache = pkl.load(f)
            except (EOFError, pkl.UnpicklingError):
                # File is empty or corrupted (e.g. crashed mid-write); start fresh
                self.embed_cache = {}
        else:
            self.embed_cache = {}

        # question to NER cache
        self.Q_NER_cache = Cache(self.Q_NER_cache_path)

        # single entity to relevant pages cache
        self.relevant_pages_cache = Cache(self.relevant_pages_cache_path)

        # prompt query to top-5 passage cache
        self.cache = Cache(self.cache_path)

    def save_cache(self):
        with self._embed_lock:
            if self.add_n_embed > 0:
                disk_cache = {}
                if os.path.exists(self.embed_cache_path):
                    try:
                        with open(self.embed_cache_path, "rb") as f:
                            disk_cache = pkl.load(f)
                    except (EOFError, pkl.UnpicklingError):
                        pass  # file empty/corrupted from prior crash; overwrite it

                disk_cache.update(self.embed_cache)  # in-memory wins over disk

                with open(self.embed_cache_path, "wb") as f:
                    pkl.dump(disk_cache, f)

    def get_topk_passages(self, topic, retrieval_query, key_passages, k=5):
        if self.encoder is None:
            self.load_encoder()

        if not key_passages:
            return []

        add_n_embed = 0
        passage_vectors_all = None
        passages_all = []

        # Snapshot the embed cache once — brief lock, no contention during encoding
        with self._embed_lock:
            embed_snapshot = dict(self.embed_cache)

        # Separate cached vs uncached topics
        topics_to_encode = {}
        topic_order = []
        cached_vectors = {}

        for t, passages in key_passages.items():
            passages_all += passages
            if t in embed_snapshot and embed_snapshot[t].shape[0] == len(passages):
                cached_vectors[t] = embed_snapshot[t]
            else:
                topics_to_encode[t] = passages
                topic_order.append(t)

        # Batch encode all missing topics in a single GPU call
        if topics_to_encode:
            all_inputs = []
            topic_boundaries = {}
            idx = 0
            for t in topic_order:
                inputs = [
                    psg["title"]
                    + " "
                    + psg["text"].replace("<s>", "").replace("</s>", "")
                    for psg in topics_to_encode[t]
                ]
                topic_boundaries[t] = (idx, idx + len(inputs))
                all_inputs.extend(inputs)
                idx += len(inputs)

            with self._encode_lock:
                all_vectors = self.encoder.encode(
                    all_inputs,
                    batch_size=self.batch_size,
                    device=self.encoder.device,
                )

            with self._embed_lock:
                for t, (start, end) in topic_boundaries.items():
                    vec = all_vectors[start:end]
                    self.embed_cache[t] = vec
                    cached_vectors[t] = vec
                    add_n_embed += 1

        # Build passage_vectors_all in key_passages iteration order
        for t in key_passages:
            vecs = cached_vectors[t]
            passage_vectors_all = (
                np.concatenate([passage_vectors_all, vecs], axis=0)
                if passage_vectors_all is not None
                else vecs
            )

        if retrieval_query in self._query_vec_cache:
            query_vectors = self._query_vec_cache.pop(retrieval_query)
        else:
            with self._encode_lock:
                query_vectors = self.encoder.encode(
                    [retrieval_query], batch_size=self.batch_size, device=self.encoder.device
                )[0]

        scores = np.inner(query_vectors, passage_vectors_all)
        indices = np.argsort(-scores)[:k]

        if add_n_embed > 0:
            self.add_n_embed += add_n_embed
            self.save_cache()

        indices = [i for i in indices if i < len(passages_all)]
        return [passages_all[i] for i in indices]

    def make_ner_cache(self, questions: List[str]):
        # Filter to uncached questions
        uncached = [q for q in questions if self.Q_NER_cache.get_item(q) is None]
        if not uncached:
            return
        
        print(f"Prewarming NER cache for {len(uncached)} questions...", flush=True)
        # Batch NER inference
        all_results = self.ner(uncached, batch_size=64)
        
        for question, ner_results in zip(uncached, all_results):
            ners = [r["word"] for r in ner_results if "#" not in r["word"]]
            self.Q_NER_cache.set_item(question, ners)

    def _collect_key_passages(self, topic, claim, question):
        """Collect the passage pool for one claim (DB + cache lookups only, no encoding)."""
        ners = self.Q_NER_cache.get_item(question) or []
        ner_relevant_titles = []
        for ner in ners:
            pgs_selected = self.relevant_pages_cache.get_item(ner)
            if pgs_selected:
                pgs = pgs_selected
            else:
                pgs = self.db.get_relevant_titles(ner)
                if not pgs:
                    continue
                self.relevant_pages_cache.set_item(ner, pgs)
            ner_relevant_titles += [
                pg
                for pg in pgs
                if ((pg.lower() in claim.lower()) or (pg.lower() in question.lower()))
            ]

        combined = [topic] + ner_relevant_titles + ners
        key_passages = {}
        for title in list(set(combined)):
            title = title.replace("_", " ")
            if title in self.not_existing_pages:
                continue
            try:
                key_passages[title] = self.db.get_text_from_title(title)
            except Exception:
                self.not_existing_pages.add(title)
        return key_passages

    def prewarm(self, claims):
        """Pre-compute all passage and query embeddings in batch before the claim loop.

        Args:
            claims: list of (topic, claim_text, question) tuples
        """
        if self.encoder is None:
            self.load_encoder()

        # Filter to claims not already in the retrieval cache
        uncached = [
            (topic, claim, question)
            for topic, claim, question in claims
            if self.cache.get_item(topic + "#" + claim.strip()) is None
        ]
        if not uncached:
            return

        # Phase 1: collect key_passages in parallel (DB/cache I/O, no GPU)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            all_key_passages = list(executor.map(
                lambda args: self._collect_key_passages(*args), uncached
            ))

        # Phase 2: batch encode ALL passages across ALL topics in one encoder call
        with self._embed_lock:
            already_cached = set(self.embed_cache.keys())

        topics_to_encode = {}
        for key_passages in all_key_passages:
            for t, passages in key_passages.items():
                if t not in already_cached and t not in topics_to_encode:
                    topics_to_encode[t] = passages

        if topics_to_encode:
            all_inputs = []
            topic_boundaries = {}  # topic -> (start, end) index into all_inputs
            idx = 0
            for t, passages in topics_to_encode.items():
                inputs = [
                    psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "")
                    for psg in passages
                ]
                topic_boundaries[t] = (idx, idx + len(inputs))
                all_inputs.extend(inputs)
                idx += len(inputs)

            all_vectors = self.encoder.encode(
                all_inputs, batch_size=self.batch_size, device=self.encoder.device
            )

            with self._embed_lock:
                for t, (start, end) in topic_boundaries.items():
                    self.embed_cache[t] = all_vectors[start:end]
                    self.add_n_embed += 1

            self.save_cache()
            self.add_n_embed = 0

        # Phase 3: batch encode all retrieval queries at once
        retrieval_queries = [topic + " " + claim.strip() for topic, claim, _ in uncached]
        query_vectors = self.encoder.encode(
            retrieval_queries, batch_size=self.batch_size, device=self.encoder.device
        )
        for (topic, claim, _), vec in zip(uncached, query_vectors):
            self._query_vec_cache[topic + " " + claim.strip()] = vec

    def get_topk_related_passages(self, topic, claim, question, k=5, use_cache=True):
        """
        NER based top-k passage retrieval.
        Reuses _collect_key_passages so the same titles discovered by prewarm
        are used here — avoiding cache misses during evaluation.
        """
        retrieval_query = topic + " " + claim.strip()
        cache_key = topic + "#" + claim.strip()

        # check cache
        cache_res = self.cache.get_item(cache_key)
        if use_cache and cache_res is not None:
            return cache_res

        # Reuse the shared passage collection logic
        key_passages = self._collect_key_passages(topic, claim, question)

        top_k_related_passages = self.get_topk_passages(
            topic, retrieval_query, key_passages, k
        )
        self.cache.set_item(cache_key, top_k_related_passages)

        self.add_n += 1
        return top_k_related_passages