(ns tfjs-cljs.core
  (:require-macros [tfjs-cljs.macros :refer [deftf]])
  (:refer-clojure :exclude [print max min]))

(def ^:private dtypes {:float32 "float32"
                       :int32 "int32"
                       :bool "bool"})

(defn tensor
  "Creates a tf.Tensor with the provided values, shape and dtype."
  ([values]
   (.tensor js/tf (clj->js values)))
  ([values shape]
   (.tensor js/tf (clj->js values) (clj->js shape))))

(defn scalar
  "Creates rank-0 tf.Tensor (scalar) with the provided value and dtype."
  ([value]
   (.scalar js/tf value))
  ([value dtype]
   (.scalar js/tf value (dtype dtypes))))

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

(defn ones
  "Creates a tf.Tensor with all elements set to 1."
  [shape]
  (.ones js/tf (clj->js shape)))

(defn random-normal
  "Creates a tf.Tensor with values sampled from a normal distribution."
  ([shape mean std-dev]
    (.randomNormal js/tf (clj->js shape) mean std-dev)))

(defn random-uniform
  "Creates a tf.Tensor with values sampled from a uniform distribution."
  ([shape] (random-uniform shape 0 1))
  ([shape minval] (random-uniform shape minval 1))
  ([shape minval maxval]
    (.randomUniform js/tf (clj->js shape) minval maxval)))

(defn zeros
  "Creates a tf.Tensor with all elements set to 0."
  [shape]
  (.zeros js/tf (clj->js shape)))


;; Tensor

(defn data-sync
  "Synchronously downloads the values from the tf.Tensor. This blocks the UI thread until the values are
  ready, which can cause performance issues."
  [tensor]
  (array-seq (.dataSync tensor)))

(deftf dispose)
(deftf print)

;; OPERATIONS

;; Arithmetic

(defn add
  "Adds two tf.Tensors element-wise, A + B. Supports broadcasting."
  [a b]
  (.add js/tf a b))

(defn sub
  "Subtracts two tf.Tensors element-wise, A - B. Supports broadcasting."
  [a b]
  (.sub js/tf a b))

(defn mul
  "Multiplies two tf.Tensors element-wise, A * B. Supports broadcasting."
  [a b]
  (.mul js/tf a b))

(defn div
  "Divides two tf.Tensors element-wise, A / B. Supports broadcasting.\n\n"
  [a b]
  (.div js/tf a b))

(defmulti pow
  "Computes the power of one tf.Tensor to another. Supports broadcasting."
  (fn [_ exp] (type exp)))

(defmethod pow js/Number [base exp]
  (js/tf.pow base (scalar exp)))
(defmethod pow js/tf.Tensor [base exp]
  (js/tf.pow base exp))


;; Basic math

(defn square
  "Computes square of x element-wise: x ^ 2"
  [x]
  (js/tf.square x))


;; Reduction

(deftf max)
(deftf mean)
(deftf min)
(deftf sum)


;;; Performance

(deftf memory)
