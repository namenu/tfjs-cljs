(ns tfjs-cljs.layers.models
  (:require [cljsjs.tfjs]
            [cljs.core.async :refer [chan put!]]))

(defn sequential
  "Creates a tf.Sequential model. A sequential model is any model where the outputs of one
  layer are the inputs to the next layer, i.e. the model topology is a simple 'stack' of layers, with
  no branching or skipping."
  ([]
   (.sequential js/tf))
  ([config]
   (.sequential js/tf (clj->js config))))

(defn load-model
  "Load a model, including its topology and optionally weights."
  [path]
  (let [c (chan)]
    (.then (.loadModel js/tf path)
           #(put! c %))
    c))

(defn summary
  "Print a text summray of the Sequential model's layers."
  [model]
  (.summary model)
  model)

(defn add
  "Adds a layer instance on top of the layer stack.

  Note: returning model itself for chaining is not implemented in the original."
  [model layer]
  (.add model layer)
  model)

(defn
  ^{:category "helper"}
  stack
  "Stacks layer instances on top of the layer stack."
  [model & layers]
  (loop [m model, [l & ls] layers]
    (if l
      (recur (add m l) ls)
      m)))

(defn compile
  "Configures and prepares the model for training and evaluation. Compiling outfits
  the model with an optimizer, loss, and/or metrics. Calling fit or evaluate on an
  un-compiled model will throw an error."
  [model config]
  (.compile model (clj->js config))
  model)

(defn predict
  "Generates output predictions for the input samples."
  ([model x]
   (.predict model x))
  ([model x config]
   (.predict model x (clj->js config))))

(defn fit
  "Trains the model for a fixed number of epochs (iterations on a dataset)."
  ([model x y]
   (.fit model x y))
  ([model x y config]
   (.fit model x y (clj->js config))))

(defn save
  "Save the configuration and/or weights of the Model."
  [model url]
  (let [c (chan)]
    (.then (.save model url)
           #(put! c %))
    c))

(defn get-layer
  "Retrieves a layer based on either its name (unique) or index.
  Indices are based on order of horizontal graph traversal (bottom-up).
  If both name and index are specified, index takes precedence."
  ([model name] (.getLayer model name))
  ([model name index] (.getLayer model name index)))
