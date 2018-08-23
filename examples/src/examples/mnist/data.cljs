(ns examples.mnist.data
  (:require [promesa.core :as p]
            [promesa.async-cljs :refer-macros [async]]
            [cljs.core.async :refer [chan put! take!]]
            [tfjs-cljs.core :as tf]))

(def IMAGE_SIZE 784)
(def NUM_CLASSES 10)
(def NUM_DATASET_ELEMENTS 65000)
(def CHUNK_SIZE 5000)

(def MNIST_IMAGES_SPRITE_PATH "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png")
(def MNIST_LABELS_PATH "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8")

(defonce dataset (atom nil))

(defn get-view [dataset-bytes-buffer img canvas ctx]
  (dotimes [i (/ NUM_DATASET_ELEMENTS CHUNK_SIZE)]
    (let [dataset-bytes-view (js/Float32Array. dataset-bytes-buffer
                                               (* i IMAGE_SIZE CHUNK_SIZE 4) ;; offset
                                               (* IMAGE_SIZE CHUNK_SIZE) ;; size
                                               )

          image-data         (do ctx
                                 (.drawImage ctx img
                                             0 (* i CHUNK_SIZE) (.-width img) CHUNK_SIZE
                                             0 0 (.-width img) CHUNK_SIZE)

                                 (.getImageData ctx 0 0 (.-width canvas) (.-height canvas)))]

      (dotimes [j (/ (.. image-data -data -length) 4)]
        (aset dataset-bytes-view j (/ (aget (.-data image-data) (* j 4)) 255)))))

  (js/Float32Array. dataset-bytes-buffer))

(defn request-image [c]
  (let [img    (js/Image.)
        canvas (.createElement js/document "canvas")]
    (doto img
      (aset "crossOrigin" "")
      (aset "onload" (fn []
                       (aset img "width" (.-naturalWidth img))
                       (aset img "height" (.-naturalHeight img))

                       (aset canvas "width" (.-width img))
                       (aset canvas "height" CHUNK_SIZE)

                       (let [dataset-bytes-buffer (js/ArrayBuffer. (* NUM_DATASET_ELEMENTS IMAGE_SIZE 4))
                             ctx                  (.getContext canvas "2d")
                             value                (get-view dataset-bytes-buffer img canvas ctx)]
                         (put! c {:type :images
                                  :data value})
                         #_(swap! dataset assoc :images value))
                       ))
      (aset "src" MNIST_IMAGES_SPRITE_PATH))))

(defn request-labels [c]
  (async
    (p/alet [response (p/await (js/fetch MNIST_LABELS_PATH))
             buffer (p/await (.arrayBuffer response))]
            (put! c {:type :labels
                     :data (js/Uint8Array. buffer)}))))

(defn ->tensors [size offset]
  (let [images             (:images @dataset)
        labels             (:labels @dataset)
        batch-images-array (js/Float32Array. (* size IMAGE_SIZE))
        batch-labels-array (js/Uint8Array. (* size NUM_CLASSES))]

    (dotimes [i size]
      (let [idx   (+ i offset)
            image (.slice images (* idx IMAGE_SIZE) (+ (* idx IMAGE_SIZE) IMAGE_SIZE))
            label (.slice labels (* idx NUM_CLASSES) (+ (* idx NUM_CLASSES) NUM_CLASSES))]
        (.set batch-images-array image (* i IMAGE_SIZE))
        (.set batch-labels-array label (* i NUM_CLASSES))))

    ; Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
    ; that we can feed it to our convolutional neural net.
    {:xs (-> batch-images-array
             (tf/tensor2d [size IMAGE_SIZE])
             (tf/reshape [size 28 28 1]))
     :ys (tf/tensor2d batch-labels-array [size NUM_CLASSES])}))

(defonce c (chan))

(defn load! [cb]
  (let [check-done #(when (and (contains? @dataset :images)
                               (contains? @dataset :labels))
                      (cb))]
    (if @dataset
      (check-done)
      (do
        (reset! dataset {})

        (request-image c)
        (request-labels c)

        (let [f (fn [{:keys [type data]}]
                  (swap! dataset assoc type data)
                  (check-done))]
          (take! c f)
          (take! c f))))))
