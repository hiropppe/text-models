import logging
import numpy as np
import time

from .base import BaseModel
from .util import (
    detect_input,
    read_corpus,
    perplexity,
    assign_random_topic,
    draw_phi,
    draw_theta,
    get_topics,
    Progress
)

from . import lda_c as lda


class LDA(BaseModel):
    
    def __init__(self,
                 corpus,
                 word2id=None,
                 K=20,
                 alpha=0.1,
                 beta=0.01,
                 n_iter=1000,
                 report_every=10):
        corpus = detect_input(corpus)

        self.K = K
        self.alpha = alpha
        self.beta = beta

        if word2id:
            self.word2id = {iw.split("\t")[0]: int(iw.split("\t")[1]) for iw in open(word2id).read().strip().split("\n")}
        else:
            self.word2id = {}

        W, self.vocab, self.word2id = read_corpus(corpus, word2id=self.word2id)
        Z = assign_random_topic(W, self.K)

        D = len(W)
        N = sum(len(d) for d in W)
        V = len(self.vocab)

        logging.info(f"Corpus: {D} docs, {N} words, {V} vocab.")
        logging.info(f"Number of topics: {self.K}")
        logging.info(f"alpha: {self.alpha:.3f}")
        logging.info(f"beta: {self.beta:.3f}")

        logging.info("Running Gibbs sampling inference")
        logging.info(f"Number of sampling iterations: {n_iter}")

        self.n_kw = np.zeros((self.K, V), dtype=np.int32)  # number of word w assigned to topic k
        self.n_dk = np.zeros((D, self.K), dtype=np.int32)  # number of words in document d assigned to topic k
        self.n_k = np.zeros((self.K), dtype=np.int32)  # total number of words assigned to topic k
        self.n_d = np.zeros((D), dtype=np.int32)  # number of word in document (document length)

        lda.init(W, Z, self.n_kw, self.n_dk, self.n_k, self.n_d)

        progress = Progress(n_iter)
        start = time.time()
        for i in range(n_iter):
            lda.inference(W, Z, self.n_kw, self.n_dk, self.n_k, self.n_d, self.alpha, self.beta)
            if i % report_every == 0:
                ppl = perplexity(N, self.n_kw, self.n_dk, self.alpha, self.beta)
            progress.update(ppl)
        elapsed = time.time() - start

        ppl = perplexity(N, self.n_kw, self.n_dk, self.alpha, self.beta)

        logging.info(f"Sampling completed! Elapsed {elapsed:.4f} sec ppl={ppl:.3f}")

    def get_topics(self, topn=10):
        return get_topics(self.n_kw, self.vocab, self.beta, topn=topn)

    @property
    def theta(self):
        return draw_theta(self.n_dk, self.alpha)

    @property
    def phi(self):
        return draw_phi(self.n_kw, self.beta)

