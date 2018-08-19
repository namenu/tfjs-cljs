(ns tfjs-cljs.layers
  (:refer-clojure :exclude [flatten])
  (:require [cljsjs.tfjs]))


;; Basic

(defn activation
  "Applies an activation function to an output."
  [activation-fn]
  (.activation js/tf.layers (clj->js {:activation activation-fn})))

(defn dense
  "Creates a dense (fully connected) layer."
  [config]
  (.dense js/tf.layers (clj->js config)))

(defn flatten
  "Flattens the input. Does not affect the batch size.

  A Flatten layer flattens each batch in its inputs to 1D (making the output 2D)."
  ([] (flatten {}))
  ([config]
   (.flatten js/tf.layers (clj->js config))))

(defn repeat-vector
  "Repeats the input n times in a new dimension."
  [config]
  (.repeatVector js/tf.layers (clj->js config)))


;; Convolutional

(defn conv1d
  "1D convolution layer (e.g., temporal convolution).

  This layer creates a convolution kernel that is convolved with the layer input over a single
  spatial (or temporal) dimension to produce a tensor of outputs."
  [config]
  (.conv1d js/tf.layers (clj->js config)))

(defn conv2d
  "2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved with the layer input to produce a
  tensor of outputs."
  [config]
  (.conv2d js/tf.layers (clj->js config)))


;; Pooling

(defn max-pooling-2d
  "Max pooling operation for spatial data."
  [config]
  (.maxPooling2d js/tf.layers (clj->js config)))


;; Recurrent

(defn simpleRNN
  "Fully-connected RNN where the output is to be fed back to input."
  [config]
  (.simpleRNN js/tf.layers (clj->js config)))


;; Wrapper

(defn time-distributed
  "This wrapper applies a layer to every temporal slice of an input.

  The input should be at least 3D, and the dimension of the index 1 will be considered to be
  the temporal dimension."
  ([layer]
    (.timeDistributed js/tf.layers (clj->js {:layer layer})))
  ([layer input-shape]
    (.timeDistributed js/tf.layers (clj->js {:layer layer :inputShape input-shape}))))
