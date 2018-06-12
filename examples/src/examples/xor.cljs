(ns examples.xor
  (:require [cljsjs.p5]
            [p5-cljs.p5-quil :as q]
            [tfjs-cljs.core :as tf :refer-macros [with-tidy with-fit-history]]
            [tfjs-cljs.models :as models]
            [tfjs-cljs.layers :as layers]
            [tfjs-cljs.train :as train]))


(defonce optimizer (train/adam 0.1))
(defonce train-xs (tf/tensor2d [[0 0] [1 0] [0 1] [1 1]]))
(defonce train-ys (tf/tensor2d [[0] [1] [1] [0]]))

(defn setup! []
  (js/createCanvas 400 400)

  (-> (models/sequential)

      ; hidden layer
      (models/add (layers/dense {:inputShape [2]
                                 :units      2
                                 :activation "sigmoid"}))

      ; output
      (models/add (layers/dense {:units      1
                                 :activation "sigmoid"}))

      (models/compile {:optimizer optimizer
                       :loss      "meanSquaredError"})))

(defn update! [model]
  (let [config {:shuffle true
                #_#_:epochs  10}]
    (with-fit-history [model train-xs train-ys config] history
      (let [loss (first (:loss history))]
        (console.log loss))))
  model)

(defn draw! [model]

  (with-tidy
    (let [resolution 40
          cols       (/ js/width resolution)
          rows       (/ js/height resolution)

          inputs     (for [x (range cols) y (range rows)]
                       [(/ x cols) (/ y cols)])

          xs         (tf/tensor2d inputs)
          ys         (models/predict model xs)

          vs         (tf/data-sync ys)]

      (doseq [[[x y] v] (zipmap (for [x (range cols) y (range rows)]
                                  [(* x resolution) (* y resolution)])
                          vs)
              :let [br (* v 255)]]
        (js/fill br)
        (js/rect x y resolution resolution)
        (js/fill (- 255 br))
        (js/textAlign js/CENTER js/CENTER)
        (js/text (js/nf v 1 2) (+ x (/ resolution 2)) (+ y (/ resolution 2))))

      )))

(defonce state (atom {}))

(q/sketch state
  :setup setup!
  :update update!
  :draw draw!)
