import logging
import numpy as np
import time

from . import atm_c as model
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


class ATM():

    def __init__(self,
                 corpus,
                 author,
                 K=20,
                 alpha=0.1,
                 beta=0.01,
                 n_iter=1000,
                 report_every=10):
        corpus = detect_input(corpus)
        author = detect_input(author)
        self.K = K
        self.alpha = alpha
        self.beta = beta

        self.W, self.vocab, self.word2id = read_corpus(corpus)
        self.A, self.authors, self.author2id = read_corpus(author)

        self.D = len(self.W)
        self.V, self.S = len(self.vocab), len(self.authors)
        self.N, self.M = sum(len(d) for d in self.W), sum(len(d) for d in self.A)

        self.Z = assign_random_topic(self.W, self.K)
        self.Y = [np.random.randint(len(self.A[d]), size=len(self.W[d])) for d in range(self.D)]

        logging.info(f"Corpus: {self.D} docs, {self.N} words, {self.V} vocab.")
        logging.info(f"Author: {self.M} words, {self.S} vocab.")
        logging.info(f"Number of topics: {self.K}")
        logging.info(f"alpha: {self.alpha:.3f}")
        logging.info(f"beta: {self.beta:.3f}")

        self.n_kw = np.zeros((self.K, self.V), dtype=np.int32)  # number of word w assigned to topic k
        self.n_dk = np.zeros((self.D, self.K), dtype=np.int32)  # number of words in document d assigned to topic k
        self.n_ak = np.zeros((self.S, self.K), dtype=np.int32)  # number of topic k generated by author y
        self.n_k = np.zeros((self.K), dtype=np.int32)  # total number of words assigned to topic k
        self.n_d = np.zeros((self.D), dtype=np.int32)  # number of word in document (document length)
        self.n_a = np.zeros((self.S), dtype=np.int32)  # total number of topics generated by author y
        self.n_ad = np.zeros((self.D), dtype=np.int32) # number of author in document (author length)

        model.init(self.W, self.A, self.Z, self.Y, self.n_kw, self.n_dk, self.n_ak, self.n_k, self.n_d, self.n_a, self.n_ad)

        logging.info("Running Gibbs sampling inference")
        logging.info(f"Number of sampling iterations: {n_iter}")

        start = time.time()
        pbar = Progress(n_iter)
        for i in range(n_iter):
            model.inference(self.W, self.A, self.Z, self.Y, self.N, self.n_kw, self.n_dk, self.n_ak, self.n_k, self.n_d, self.n_a, self.n_ad, self.alpha, self.beta)
            if i % report_every == 0:
                ppl = perplexity(self.N, self.n_kw, self.n_dk, self.alpha, self.beta)
            pbar.update(ppl)

        elapsed = time.time() - start
        ppl = perplexity(self.N, self.n_kw, self.n_dk, self.alpha, self.beta)
        logging.info(f"Sampling completed! Elapsed {elapsed:.4f} sec ppl={ppl:.3f}")

        self.__params = {}
        self.__params['theta'] = draw_theta(self.n_dk, self.beta)
        self.__params['phi'] = draw_phi(self.n_kw, self.alpha)
    
    def get_topics(self, topn=10):
        return get_topics(self.n_kw, self.vocab, self.beta, topn=topn)
    
    def __getitem__(self, key):
        return self.__params[key]
