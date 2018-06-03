(ns tfjs-cljs.core)

(defn tensor
  "Creates a tf.Tensor with the provided values, shape and dtype."
  ([values]
   (.tensor js/tf (clj->js values)))
  ([values shape]
   (.tensor js/tf (clj->js values) (clj->js shape))))

(defn scalar
  "Creates rank-0 tf.Tensor (scalar) with the provided value and dtype."
  ([value]
   (.scalar js/tf (clj->js value)))
  ([value dtype]
   (.scalar js/tf (clj->js value) dtype)))

(defn tensor1d
  "Creates rank-1 tf.Tensor with the provided values, shape and dtype."
  ([values]
   (.tensor1d js/tf (clj->js values)))
  ([values dtype]
   (.tensor1d js/tf (clj->js values) (clj->js dtype))))

(defn tensor2d
  "Creates rank-2 tf.Tensor with the provided values, shape and dtype."
  ([values]
   (.tensor2d js/tf (clj->js values)))
  ([values [x y]]
   (.tensor2d js/tf (clj->js values) (array x y))))

(defn tensor3d
  "Creates rank-3 tf.Tensor with the provided values, shape and dtype."
  ([values]
   (.tensor3d js/tf (clj->js values)))
  ([values [x y z]]
   (.tensor3d js/tf (clj->js values) (array x y z))))

(defn variable
  "Creates a new variable with the provided initial value."
  [initialValue]
  (.variable js.tf initialValue))

(defn zeros
  "Creates a tf.Tensor with all elements set to 0."
  [shape]
  (.zeros js/tf (clj->js shape)))

(defn ones
  "Creates a tf.Tensor with all elements set to 1."
  [shape]
  (.ones js/tf (clj->js shape)))
