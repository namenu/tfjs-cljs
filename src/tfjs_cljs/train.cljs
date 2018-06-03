(ns tfjs-cljs.train)

(defn sgd
  "Constructs a tf.SGDOptimizer that uses stochastic gradient descent."
  [learning-rate]
  (.sgd js/tf.train learning-rate))

(defn minimize
  "Executes f() and minimizes the scalar output of f() by computing gradients of y with respect to the
  list of trainable variables provided by varList. If no list is provided, it defaults to all trainable
  variables."
  [optimizer f]
  (.minimize optimizer f))
