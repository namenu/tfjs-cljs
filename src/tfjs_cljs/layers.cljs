(ns tfjs-cljs.layers)

(defn simpleRNN
  "Fully-connected RNN where the output is to be fed back to input."
  [config]
  (.simpleRNN js/tf.layers (clj->js config)))
