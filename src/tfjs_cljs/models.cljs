(ns tfjs-cljs.models)

(defn add
  "Adds a layer instance on top of the layer stack."
  [model layer]
  (.add model layer))

(defn compile
  "Configures and prepares the model for training and evaluation. Compiling outfits
  the model with an optimizer, loss, and/or metrics. Calling fit or evaluate on an
  un-compiled model will throw an error."
  [model config]
  (.compile model (clj->js config)))
