# tfjs-cljs

A ClojureScript wrapper library for [TensorFlow.js](https://js.tensorflow.org/).

## Usage

Add a dependency to your `project.clj`.

```
[namenu/tfjs-cljs "0.1.0-SNAPSHOT"]
```

## Example

Clojure idiomatic rewrite of linear regression example from [#getting-started](https://js.tensorflow.org/#getting-started).

```clojure
(let [model (models/sequential)]
  (-> model
      (models/add (layers/dense {:units 1, :inputShape [1]}))
      (models/compile {:loss "meanSquaredError", :optimizer "sgd"}))

  (let [xs (tf/tensor2d [1 2 3 4] [4 1])
        ys (tf/tensor2d [1 3 5 7] [4 1])]
    (.then (models/fit model xs ys)
           #(.print (models/predict model (tf/tensor2d [5] [1 1]))))))
```

For memory management, `with-tidy` macro might come in handy.

```clojure
(:require [tfjs-cljs.core :as tf :refer-macros [defvar with-tidy]]
          [tfjs-cljs.train :as train])

; y = a * x + b
(defvar a (tf/scalar (q/random 1)))
(defvar b (tf/scalar (q/random 1)))

(defn predict [x]
  (let [xs (tf/tensor1d x)
        ys (.. xs (mul a) (add b))]
    ys))

(defn solve [xs ys]
  (tf/with-tidy
    (train/minimize optimizer
                    #(loss (predict xs) ys))))
```


## Should I use it?

No. This library only contains a subset of `TensorFlow.js` to the extent of my knowledge.
