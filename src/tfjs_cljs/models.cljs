(ns tfjs-cljs.models)

(defn sequential
  "Creates a tf.Sequential model. A sequential model is any model where the outputs of one
  layer are the inputs to the next layer, i.e. the model topology is a simple 'stack' of layers, with
  no branching or skipping."
  ([]
   (.sequential js/tf))
  ([config]
   (.sequential js/tf (clj->js config))))

(defn add
  "Adds a layer instance on top of the layer stack.

  Note: returning model itself for chaining is not implemented in the original."
  [model layer]
  (.add model layer)
  model)

(defn compile
  "Configures and prepares the model for training and evaluation. Compiling outfits
  the model with an optimizer, loss, and/or metrics. Calling fit or evaluate on an
  un-compiled model will throw an error."
  [model config]
  (.compile model (clj->js config)))

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
