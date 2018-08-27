# tfjs-cljs

A ClojureScript wrapper library for [TensorFlow.js](https://js.tensorflow.org/).

## Usage

Add a dependency to your `project.clj`.

[![Clojars Project](https://img.shields.io/clojars/v/namenu/tfjs-cljs.svg)](https://clojars.org/namenu/tfjs-cljs)

## Example

Clojure idiomatic rewrite of linear regression example from [#getting-started](https://js.tensorflow.org/#getting-started).

```clojure
(def model
  (-> (models/sequential)
      (models/add (layers/dense {:units 1 :inputShape [1]}))
      (models/compile {:loss "meanSquaredError" :optimizer "sgd"})))

(let [xs     (tf/tensor2d [1 2 3 4] [4 1])
      ys     (tf/tensor2d [1 3 5 7] [4 1])]
  (.then (models/fit model xs ys config {:epochs 10})
         #(tf/print (models/predict model (tf/tensor2d [5] [1 1])))))
```

See more (examples)[examples].

```clojure
(def model
  (-> (models/sequential)
      (models/stack (layers/dense {:units 100 :activation "relu" :inputShape [10]})
                    (layers/dense {:units 1 :activation "linear"}))
      (models/compile {:optimizer "sgd" :loss "meanSquaredError"})))

(let [xs     (tf/random-normal [100 10])
      ys     (tf/random-normal [100 1])
      config {:epochs    100
              :callbacks {:onEpochEnd
                          (fn [epoch log]
                            (let [loss (aget log "loss")]
                              (.log js/console (str "Epoch" epoch ": loss = " loss))))}}]
  (models/fit model xs ys config))
```


## Should I use it?

No. This library only contains a subset of `TensorFlow.js` to the extent of my knowledge.
