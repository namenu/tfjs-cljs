# tfjs-cljs

A ClojureScript wrapper library for [TensorFlow.js](https://js.tensorflow.org/).

## Usage

Add a dependency to your `project.clj`.

```
[namenu/tfjs-cljs "0.1.0-SNAPSHOT"]
```

## Example

Following example gets a regression line for given input.

```clojure
(:require [tfjs-cljs.core :as tf])

; y = a * x + b
(defvar a (tf/scalar (q/random 1)))
(defvar b (tf/scalar (q/random 1)))

(defn predict [x]
  (let [xs (tf/tensor1d x)
        ys (.. xs (mul a) (add b))]
    ys))

(defn solve [xs ys]
  (tf/with-tidy
    (when-not (empty? xs)
      (let [ys (tf/tensor1d ys)]
        (train/minimize optimizer #(loss (predict xs) ys))))))

(defn regression-line [xs]
  (tf/with-tidy
    (let [ys (predict xs)]
      (.dataSync ys))))
```


## License

Copyright © 2018 Hyunwoo Nam

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
