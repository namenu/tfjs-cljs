(ns tfjs-cljs.core)

(defn scalar
  "Creates rank-0 tf.Tensor (scalar) with the provided value and dtype."
  ([value]
    (.scalar js/tf (clj->js value)))
  ([value dtype]
    (.scalar js/tf (clj->js value) (clj->js dtype))))

(defn tensor1d
  "Creates rank-1 tf.Tensor with the provided values, shape and dtype."
  ([values]
    (.tensor1d js/tf (clj->js values)))
  ([values dtype]
    (.tensor1d js/tf (clj->js values) (clj->js dtype))))
