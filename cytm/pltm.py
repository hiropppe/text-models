import logging
import numpy as np
import time

import pandas as pd

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

from . import pltm_c as model


class PLTM(BaseModel):

    def __init__(self,
                 *corpuses,
                 K=20,
                 alpha=0.1,
                 beta=0.01,
                 n_iter=1000,
                 report_every=10):

        self.K = K
        self.alpha = alpha

        T = len(corpuses)
        W_t, self.vocab_t, self.word2id_t = [], [], []
        Z_t = []
        V_t, N_t = [], []
        for t, corpus in enumerate(corpuses):
            if isinstance(corpus, tuple) or isinstance(corpus, list):
                corpus, word2id = corpus[0], corpus[1]
            else:
                word2id = None

            corpus = detect_input(corpus)

            if word2id:
                word2id = {iw.split("\t")[0]: int(iw.split("\t")[1]) for iw in open(word2id).read().strip().split("\n")}
            else:
                word2id = {}

            W, vocab, word2id = read_corpus(corpus, word2id=word2id)
            Z = assign_random_topic(W, K)

            V = len(vocab)
            N = sum(len(d) for d in W)

            W_t.append(W)
            self.vocab_t.append(vocab)
            self.word2id_t.append(word2id)
            Z_t.append(Z)
            V_t.append(V)
            N_t.append(N)

        D = len(W_t[0])
        self.beta = np.array([beta]*T)

        logging.info(f"Corpus: {D} docs, {N_t[0]} words, {V_t[0]} vocab.")

        for t in range(1, T):
            logging.info(f"Side Information[{t-1}]: {N_t[t]} words, {V_t[t]} vocab.")

        logging.info(f"Number of topics: {K}")
        logging.info(f"alpha: {alpha:.3f}")
        logging.info(f"beta: {beta:.3f}")

        self.n_kw = [np.zeros((self.K, V_t[t]), dtype=np.int32) for t in range(T)]  # number of word w assigned to topic k
        self.n_dk = np.zeros((T, D, self.K), dtype=np.int32)  # number of word in document d assigned to topic k
        self.n_k = np.zeros((T, self.K), dtype=np.int32)  # total number of words assigned to topic k
        self.n_d = np.zeros((T, D), dtype=np.int32)  # number of word in document (document length)

        model.init(W_t, Z_t, self.n_kw, self.n_dk, self.n_k, self.n_d)

        logging.info("Running Gibbs sampling inference")
        logging.info(f"Number of sampling iterations: {n_iter}")

        start = time.time()
        pbar = Progress(n_iter)
        for i in range(n_iter):
            model.inference(W_t, Z_t, N_t, self.n_kw, self.n_dk, self.n_k, self.n_d, self.alpha, self.beta)
            if i % report_every == 0:
                ppl = perplexity(N_t[0], self.n_kw[0], self.n_dk[0], self.alpha, self.beta[0])
            pbar.update(ppl)
        
        elapsed = time.time() - start
        ppl = perplexity(N_t[0], self.n_kw[0], self.n_dk[0], self.alpha, self.beta[0])
        logging.info(f"Sampling completed! Elapsed {elapsed:.4f} sec ppl={ppl:.3f}")

    def get_topics(self, topn=10):
        return get_topics(self.n_kw[0], self.vocab_t[0], self.beta, topn=topn)

    @property
    def theta(self):
        return draw_theta(self.n_dk[0], self.alpha)

    @property
    def phi(self):
        return [draw_phi(self.n_kw[t], self.beta[t]) for t in range(len(self.n_kw))]

