(ns tfjs-cljs.train
  (:require [cljsjs.tfjs]))

(defn sgd
  "Constructs a tf.SGDOptimizer that uses stochastic gradient descent."
  [learning-rate]
  (.sgd js/tf.train learning-rate))

(defn momentum
  "Constructs a tf.MomentumOptimizer that uses momentum gradient descent."
  [learning-rate momentum]
  (.momentum js/tf.train learning-rate momentum))

(defn adagrad
  "Constructs a tf.AdagradOptimizer that uses the Adagrad algorithm."
  ([learning-rate]
   (.adagrad js/tf learning-rate))
  ([learning-rate initial-accumulator-value]
   (.adagrad js/tf learning-rate initial-accumulator-value)))

(defn adam
  "Constructs a AdamOptimizer that uses the Adam algorithm."
  [& args]
  (apply js/tf.train.adam args))

(defn minimize
  "Executes f() and minimizes the scalar output of f() by computing gradients of y with respect to the
  list of trainable variables provided by varList. If no list is provided, it defaults to all trainable
  variables."
  [optimizer f]
  (.minimize optimizer f))


(def losses (js->clj js/tf.losses :keywordize-keys true))
