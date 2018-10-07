(ns examples.mnist.main
  (:require [cljsjs.p5]
            [reagent.core :as r]
            [examples.mnist.ui :as ui]
            [examples.mnist.data :as data]
            [tfjs-cljs.core.train :as train]
            [tfjs-cljs.core :as tf :refer-macros [with-tidy]]
            [tfjs-cljs.layers :as layers]
            [tfjs-cljs.layers.models :as models]))

(def BATCH_SIZE 64)
(def TRAIN_BATCHES 150)
(def NUM_EPOCHS 1)

(def PREDICT_SIZE 100)

(def LEARNING_RATE 0.15)
(def optimizer (train/sgd LEARNING_RATE))

(defn create-model []
  #_(-> (models/sequential)
        (models/stack (layers/conv2d {:inputShape        [28 28 1]
                                      :kernelSize        5
                                      :filters           8
                                      :strides           1
                                      :activation        "relu"
                                      :kernelInitializer "varianceScaling"})

                      (layers/max-pooling-2d {:poolSize [2 2]
                                              :strides  [2 2]})

                      (layers/conv2d {:kernelSize        5
                                      :filters           16
                                      :strides           1
                                      :activation        "relu"
                                      :kernelInitializer "varianceScaling"})

                      (layers/max-pooling-2d {:poolSize [2 2]
                                              :strides  [2 2]})

                      (layers/flatten)

                      (layers/dense {:units             10
                                     :kernelInitializer "varianceScaling"
                                     :activation        "softmax"}))

        (models/compile {:optimizer optimizer
                         :loss      "categoricalCrossentropy"
                         :metrics   ["accuracy"]})

        (models/summary))

  (-> (models/sequential)
      (models/stack (layers/conv2d {:inputShape [28 28 1]
                                    :filters    32
                                    :kernelSize 3
                                    :activation "relu"})
                    (layers/conv2d {:filters    32
                                    :kernelSize 3
                                    :activation "relu"})
                    (layers/max-pooling-2d {:poolSize [2 2]})
                    (layers/conv2d {:filters    64
                                    :kernelSize 3
                                    :activation "relu"})
                    (layers/conv2d {:filters    64
                                    :kernelSize 3
                                    :activation "relu"})
                    (layers/max-pooling-2d {:poolSize [2 2]})
                    (layers/flatten)
                    (layers/dropout {:rate 0.25})
                    (layers/dense {:units      512
                                   :activation "relu"})
                    (layers/dropout {:rate 0.5})
                    (layers/dense {:units      10
                                   :activation "softmax"}))

      (models/compile {:optimizer "rmsprop"
                       :loss      "categoricalCrossentropy"
                       :metrics   ["accuracy"]})))

(defn on-batch-end [_ logs]
  (ui/update-loss logs)
  (ui/update-accuracy logs)

  (tf/next-frame))

(defn train [model]
  (ui/set-status! "Training...")
  (ui/reset-data!)

  (let [{:keys [xs ys]} (data/->tensors (* BATCH_SIZE TRAIN_BATCHES) 0)]
    (models/fit model xs ys {:batchSize       BATCH_SIZE
                             :validationSplit 0.10
                             :shuffle         true
                             :epochs          NUM_EPOCHS
                             :callbacks       {:onBatchEnd on-batch-end
                                               :onTrainEnd #(do
                                                              (ui/set-status! "Trained.")
                                                              (tf/dispose xs)
                                                              (tf/dispose ys))}})))

(defn predict [model]
  (ui/set-status! "Testing...")
  (ui/reset-predictions)

  (with-tidy [{:keys [xs ys]} (data/->tensors PREDICT_SIZE 0)
              output (models/predict model xs)

              axis 1
              labels (tf/data-sync (tf/arg-max ys axis))
              predictions (tf/data-sync (tf/arg-max output axis))]
    (dotimes [i PREDICT_SIZE]
      (let [image (tf/slice xs [i 0] [1 (aget xs "shape" 1)])
            data  (tf/data-sync (tf/flatten image))

            label (nth labels i)
            pred  (nth predictions i)]
        (ui/show-test-results data label pred)))))

(defn load-data []
  (ui/set-status! "Loading...")

  (let [fn #(ui/set-status! "Loaded!")]
    (data/load! fn)))

(defonce model (atom nil))

(defn app []
  [:div
   [:button {:on-click load-data} "Load data"]
   [:button {:on-click #(do
                          (reset! model (create-model))
                          (train @model))} "Train!"]
   [:button {:on-click #(predict @model)} "Predict"]])

(defn ^export run []
  (r/render [app]
            (js/document.getElementById "app")))

(run)
