(ns examples.mobilenet.main
  (:require [reagent.core :as r]
            [tfjs-cljs.core :as tf :refer-macros [with-tidy]]
            [tfjs-cljs.layers.models :as models]
            [cljs.core.async :refer [<!] :refer-macros [go]]))

(def MOBILENET_MODEL_PATH "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json")

(def IMAGE_SIZE 224)
(def TOPK_PREDICTIONS 10)

(defonce mobilenet (r/atom nil))

(defn load-model []
  (console.log "Loading model...")
  (go
    (let [model (<! (models/load-model MOBILENET_MODEL_PATH))]
      ; warmup
      (tf/dispose (models/predict model (tf/zeros [1 IMAGE_SIZE IMAGE_SIZE 3])))

      (console.log "Loaded.")

      (reset! mobilenet model))))

(defn get-top-k-classes [logits top-k]
  (let [values (tf/data-sync logits)

        sorted (->> (map vector values (range))
                    (sort-by #(- (first %)))
                    (take top-k)

                    (map (fn [[val idx]]
                           [(aget js/IMAGENET_CLASSES idx) val])))]
    sorted))

(defn show-results [img-el classes]
  (let [pred-container  (doto (.createElement js/document "div")
                          (aset "className" "pred-container"))
        img-container   (doto (.createElement js/document "div")
                          (.appendChild img-el))
        probs-container (.createElement js/document "div")]

    (doseq [[name prob] classes
            :let [class-el (doto (.createElement js/document "div")
                             (aset "className" "cell")
                             (aset "innerText" name))
                  probs-el (doto (.createElement js/document "div")
                             (aset "className" "cell")
                             (aset "innerText" (.toFixed prob 3)))

                  row      (doto (.createElement js/document "div")
                             (aset "className" "row")
                             (.appendChild class-el)
                             (.appendChild probs-el))]]
      (.appendChild probs-container row))

    (doto pred-container
      (.appendChild img-container)
      (.appendChild probs-container))

    (-> (.getElementById js/document "predictions")
        (.appendChild pred-container))))

(defn predict [model img-element]
  (console.log "Predicting...")

  (let [logits
        (with-tidy [img (.toFloat (tf/from-pixels img-element))
                    offset (tf/scalar 127.5)

                    ; Normalize the image from [0, 255] to [-1, 1]
                    normalized (tf/div (tf/sub img offset) offset)

                    batched (tf/reshape normalized [1 IMAGE_SIZE IMAGE_SIZE 3])]
                   (models/predict model batched))]

    (show-results img-element (get-top-k-classes logits TOPK_PREDICTIONS))))

(when-not @mobilenet
  (load-model))

(defn app []
  [:div
   [:h3 "TensorFlow.js: Using a pretrained MobileNet"]
   [:p "This demo uses the pretrained MobileNet_25_224 model from Keras which you can find "
    [:a {:href "https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_2_5_224_tf.h5"}
     "here"]
    ". It is not trained to recognize human faces. For best performance, upload images of objects
    like piano, coffee mugs, bottles, etc."]

   (when @mobilenet
     [:button {:on-click #(let [el (.getElementById js/document "cat")]
                            (predict @mobilenet el))}
      "Run!"])

   [:div {:id    "file-container"
          :style {:display "none"}}
    "Upload an image: "
    [:input {:type     "file"
             :id       "files"
             :name     "files[]"
             :multiple ""}]]

   [:div {:id "status"}]
   [:div {:id "predictions"}]

   [:img {:style  {:display "visible"}
          :id     "cat"
          :src    "cat.jpg"
          :width  224
          :height 224}]
   ])

(defn ^export run []
  (r/render [app]
            (js/document.getElementById "app")))

(run)
