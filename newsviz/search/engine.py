# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:44:48 2021

@author: Sidorenko D
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Ranker():
  def init(self, embs, meta):
    """
    embs: numpy array of vectors for each document
    meta: list of arbitrary metainformation with same indexing (used for output) == sample
    """
    self.embs = embs
    self.meta = meta
    #
    self.model = None

  def get_nearest(self, v_query, topn=10, metric='cosine'):
    """
    Outputs indexes of topn nearest vectors from self.embs
    v_query: vector of the query
    topn: how many indexes to output
    metric: see sklearn docs for NearestNeighbors
    """
    self.v_query = v_query
    if self.model is None:
      self.model = NearestNeighbors(metric=metric, n_jobs=-1)
      self.model.fit(self.embs)

    ixs = self.model.kneighbors(self.v_query)
    return ixs
  #
  def get_attributes(self, ixs):
    attrs = ixs[0]
    return attrs

