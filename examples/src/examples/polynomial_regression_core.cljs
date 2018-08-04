(ns examples.polynomial-regression-core
  (:require [tfjs-cljs.core :as t :refer-macros [defvar with-tidy]]
            [tfjs-cljs.core.train :as train]
            [examples.p5 :as p5]))

(defonce vars
  (atom {:a (t/variable (t/scalar (Math.random)))
         :b (t/variable (t/scalar (Math.random)))
         :c (t/variable (t/scalar (Math.random)))
         :d (t/variable (t/scalar (Math.random)))}))

(def num-iterations 75)
(def learning-rate 0.2)
(defonce optimizer (train/sgd learning-rate))

(defn predict [x]
  (with-tidy
    (let [ax3 (t/mul (:a @vars) (t/pow x 3) )
          bx2 (t/mul (:b @vars) (t/pow x 2) )
          cx  (t/mul (:c @vars) x)
          d   (:d @vars)]
      (-> ax3
          (t/add bx2)
          (t/add cx)
          (t/add d)))))

(defn loss [predictions labels]
  (let [mean-square-error (-> (t/sub predictions labels)
                              (t/square)
                              (t/mean))]
    mean-square-error))

(defn generate-data
  ([num-points coeff]
   (generate-data num-points coeff 0.04))
  ([num-points coeff sigma]
   (with-tidy
     (let [[a b c d] (map t/scalar coeff)

           xs (t/random-uniform [num-points] -1 1)
           ys (-> (t/mul a (t/pow xs 3))
                  (t/add (t/mul b (t/square xs)))
                  (t/add (t/mul c xs))
                  (t/add d)
                  (t/add (t/random-normal [num-points] 0 sigma)))
           ymin (t/min ys)
           ymax (t/max ys)
           yrange (t/sub ymax ymin)

           ys-normalized (-> ys
                             (t/sub ymin)
                             (t/div yrange))
           ]
       {:xs xs
        :ys ys-normalized}))))

(defn setup! []
  (js/createCanvas 400 400)

  (let [true-coefficients [-0.8 -0.2 0.9 0.5]]
    (generate-data 100 true-coefficients)))

(defn fit [{:keys [xs ys] :as state}]
  (dotimes [_ num-iterations]
    (train/minimize optimizer #(loss (predict xs) ys)))
  state)

(defn draw! [{:keys [xs ys]}]
  (js/background 0)

  (js/stroke 255)
  (js/strokeWeight 4)

  ;; draw points
  (doseq [[x y] (map vector (t/data-sync xs) (t/data-sync ys))]
    (let [px (p5/map-range x -1 1 0 js/width)
          py (p5/map-range y 0 1 js/height 0)]
      (js/point px py)))

  ;; draw regression line
  (let [xs    (range -1 1.05 0.05)
        xs-tf (t/tensor1d xs)
        ys-tf (predict xs-tf)
        ys    (t/data-sync ys-tf)]
    (js/noFill)
    (js/strokeWeight 4)
    (js/beginShape)
    (doseq [[x y] (map vector xs ys)]
      (let [x (p5/map-range x -1 1 0 js/width)
            y (p5/map-range y 0 1 js/height 0)]
        (js/vertex x y)))
    (js/endShape)

    (t/dispose xs-tf)
    (t/dispose ys-tf)))

(defonce state (atom {}))

(p5/sketch state
  :setup setup!
  :update fit
  :draw draw!
  :mouse-pressed identity)

(defn on-js-reload [])
