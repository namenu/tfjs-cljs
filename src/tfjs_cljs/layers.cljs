(ns tfjs-cljs.layers)

(defn dense
  "Creates a dense (fully connected) layer."
  [config]
  (.dense js/tf.layers (clj->js config)))

(defn simpleRNN
  "Fully-connected RNN where the output is to be fed back to input."
  [config]
  (.simpleRNN js/tf.layers (clj->js config)))
