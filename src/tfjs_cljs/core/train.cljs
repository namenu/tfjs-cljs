(ns tfjs-cljs.core.train
  (:require [cljsjs.tfjs])
  (:require-macros [tfjs-cljs.macros :refer [defconst]]))

;;; TRAINING

;; Optimizers

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

;; tf.train.Optimizer

(defn minimize
  "Executes f() and minimizes the scalar output of f() by computing gradients of y with respect to the
  list of trainable variables provided by varList. If no list is provided, it defaults to all trainable
  variables."
  [optimizer f]
  (.minimize optimizer f))


;; TODO: move to layers?

(defconst losses [:absolute-difference :compute-weighted-loss :cosine-distance :hinge-loss :huber-loss :log-loss :mean-squared-error :softmax-cross-entropy])
