(ns examples.polynomial-regression-core
  (:require [tfjs-cljs.core :as tf :refer-macros [defvar with-tidy]]
            [tfjs-cljs.train :as train]))

(def num-iterations 75)
(def learning-rate 0.5)
(defonce optimizer (train/sgd learning-rate))

(defonce vars
  (atom {:a (tf/variable (tf/scalar (Math.random)))
         :b (tf/variable (tf/scalar (Math.random)))
         :c (tf/variable (tf/scalar (Math.random)))
         :d (tf/variable (tf/scalar (Math.random)))}))

(defn predict [x]
  (with-tidy
    (let [ax3 (tf/mul (:a @vars) (tf/pow x 3) )
          bx2 (tf/mul (:b @vars) (tf/pow x 2) )
          cx  (tf/mul (:c @vars) x)
          d   (:d @vars)]
      (-> ax3
        (tf/add bx2)
        (tf/add cx)
        (tf/add d)))))

(defn loss [predictions labels]
  (let [mean-square-error (-> (tf/sub predictions labels)
                              (tf/square)
                              (tf/mean))]
    mean-square-error))

(defn train
  [xs ys]
  (let [f #(let [preds-ys (predict xs)]
             (loss preds-ys ys))]
    (dotimes [_ num-iterations]
      (train/minimize optimizer f))))

(defn generate-data
  ([num-points coeff]
   (generate-data num-points coeff 0.04))
  ([num-points coeff sigma]
   (with-tidy
     (let [[a b c d] (map tf/scalar coeff)

           xs (tf/random-uniform [num-points] -1 1)
           ys (-> (tf/mul a (tf/pow xs 3))
                  (tf/add (tf/mul b (tf/square xs)))
                  (tf/add (tf/mul c xs))
                  (tf/add d)
                  (tf/add (tf/random-normal [num-points] 0 sigma)))
           ymin (tf/min ys)
           ymax (tf/max ys)
           yrange (tf/sub ymax ymin)

           ys-normalized (-> ys
                           (tf/sub ymin)
                           (tf/div yrange))
           ]
       {:xs xs
        :ys ys-normalized}))))

(defn learn-coefficients []
  (let [true-coefficients [-0.8 -0.2 0.9 0.5]
        {:keys [xs ys]}   (generate-data 100 true-coefficients)]
    ; plot original data
    ,,,

    (map @vars tf/data-sync)
    (let [predictions-before (predict xs)]
      ; plot before train
      (console.log predictions-before)
      (tf/dispose predictions-before))

    (train xs ys)

    (map @vars tf/data-sync)
    (let [predictions-after (predict xs)]
      ; plot after train
      (console.log predictions-after)
      (tf/dispose predictions-after))

    (tf/dispose xs)
    (tf/dispose ys)))
