(ns tfjs-cljs.core
  (:refer-clojure :exclude [print max min get-in flatten keep])
  (:require [cljsjs.tfjs]
            [cljs.core.async :refer [chan put!]])
  (:require-macros [tfjs-cljs.macros :refer [deftf defconst]]))

(def ^:private dtypes {:float32 "float32"
                       :int32 "int32"
                       :bool "bool"})


;;; TENSORS

;; Creation

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

(defn buffer
  "Creates an empty tf.TensorBuffer with the specified shape and dtype."
  [shape]
  (.buffer js/tf (clj->js shape)))

(defn from-pixels
  "Create a tf.Tensor from an image."
  [pixels]
  (.fromPixels js/tf pixels))

(defn linspace
  "Return an evenly spaced sequence of numbers over the given interval."
  [start stop num]
  (.linspace js/tf start stop num))

(defn one-hot
  "Creates a one-hot tf.Tensor. The locations represented by indices take value onValue
  (defaults to 1), while all other locations take value offValue (defaults to 0)."
  [indicies depth]
  (if (sequential? indicies)
    (let [i-tensor  (tensor1d indicies "int32")
          oh-tensor (.oneHot js/tf i-tensor depth)]
      (.dispose i-tensor)
      oh-tensor)
    (.oneHot js/tf indicies depth)))

(defn ones
  "Creates a tf.Tensor with all elements set to 1."
  [shape]
  (.ones js/tf (clj->js shape)))

(defn print
  "Prints information about the tf.Tensor including its data."
  [x]
  (.print js/tf x))

(defn variable
  "Creates a new variable with the provided initial value."
  [initialValue]
  (.variable js/tf initialValue))

(defn zeros
  "Creates a tf.Tensor with all elements set to 0."
  [shape]
  (.zeros js/tf (clj->js shape)))


;; Classes
;; tf.Tensor

(defn flatten
  "Flatten a Tensor to a 1D array."
  [tensor]
  (.flatten tensor))

(defn as-scalar
  "Converts a size-1 tf.Tensor to a tf.Scalar."
  [tensor]
  (.asScalar tensor))

(defn as1d
  "Converts a tf.Tensor to a tf.Tensor1D."
  [tensor]
  (.as1D tensor))

(defn as2d
  "Converts a tf.Tensor to a tf.Tensor2D."
  [tensor rows columns]
  (.as2D tensor rows columns))

(defn as3d
  "Converts a tf.Tensor to a tf.Tensor3D."
  [tensor rows columns depth]
  (.as3D tensor rows columns depth))

(defn data
  "Asynchronously downloads the values from the tf.Tensor. Returns a promise of TypedArray that
  resolves when the computation has finished."
  [tensor]
  (let [c (chan)]
    (.then (.data tensor)
           #(put! c (array-seq %)))
    c))

(defn data-sync
  "Synchronously downloads the values from the tf.Tensor. This blocks the UI thread until the values are
  ready, which can cause performance issues."
  [tensor]
  (array-seq (.dataSync tensor)))

;; TensorBuffer

(defn get-in [buffer locs]
  (apply js-invoke buffer "get" locs))

(defn assoc-in! [buffer locs value]
  (apply js-invoke buffer "set" value locs)
  buffer)

(defn buffer->tensor
  "Creates an immutable tf.Tensor object from the buffer."
  [buffer]
  (.toTensor buffer))


;; Transformations

(defn reshape
  "Reshapes a tf.Tensor to a given shape.

  Given a input tensor, returns a new tensor with the same values as the input tensor with
  shape shape."
  [x shape]
  (.reshape js/tf x (clj->js shape)))


;; Slicing and Joining

(defn slice
  "Extracts a slice from a tf.Tensor starting at coordinates begin and is of size size."
  ([x begin]
   (.slice js/tf x (clj->js begin)))
  ([x begin size]
   (.slice js/tf x (clj->js begin) (clj->js size))))

(defn split
  "Splits a tf.Tensor into sub tensors."
  ([x num-splits] (split x num-splits 0))
  ([x num-splits axis]
   (.split js/tf x (clj->js num-splits) axis)))

;; Random

(defn random-normal
  "Creates a tf.Tensor with values sampled from a normal distribution."
  ([shape]
   (random-normal shape 0 1))
  ([shape mean]
   (random-normal shape mean))
  ([shape mean std-dev]
   (.randomNormal js/tf (clj->js shape) mean std-dev)))

(defn random-uniform
  "Creates a tf.Tensor with values sampled from a uniform distribution."
  ([shape] (random-uniform shape 0 1))
  ([shape minval] (random-uniform shape minval 1))
  ([shape minval maxval]
    (.randomUniform js/tf (clj->js shape) minval maxval)))


;;; OPERATIONS

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

(deftf arg-max)
(deftf arg-min)
(deftf max)
(deftf mean)
(deftf min)
(deftf sum)


;;; PERFORMANCE

(defn dispose
  "Disposes any tf.Tensors found within the provided object."
  [container]
  (js/tf.dispose container))

(defn keep
  "Keeps a tf.Tensor generated inside a tf.tidy() from being disposed automatically."
  [result]
  (js/tf.keep result))

(defn memory []
  (js->clj (js/tf.memory) :keywordize-keys true))

(defn next-frame []
  (js/tf.nextFrame))


;;; METRICS

(defconst metrics [:binary-accuracy :binary-crossentropy :categorical-accuracy :categorical-crossentropy :cosine-proximity :mean-squared-error])
