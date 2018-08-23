(ns examples.mnist.ui
  (:require [cljsjs.vega-embed]
            [goog.string :as gstring]
            [goog.string.format]))

(defn set-status! [msg]
  (aset (.getElementById js/document "status") "innerText" msg))

(defn- plot-loss [loss-values]
  (js/vegaEmbed "#lossCanvas"
                (clj->js {:$schema  "https://vega.github.io/schema/vega-lite/v2.json"
                          :data     {:values loss-values}
                          :mark     {:type "line"}
                          :width    260
                          :orient   "vertical"
                          :encoding {:x     {:field :batch, :type "ordinal"}
                                     :y     {:field :loss, :type "quantitative"}
                                     :color {:field :set, :type "nominal" "legend" nil}}})
                (clj->js {:width   360
                          :actions false}))

  (let [last-loss (:loss (last loss-values))]
    (-> (.getElementById js/document "loss-label")
        .-innerText
        (set! (str "last loss: " last-loss)))))

(defn- plot-accuracy [accuracy-values]
  (js/vegaEmbed "#accuracyCanvas"
                (clj->js {:$schema  "https://vega.github.io/schema/vega-lite/v2.json"
                          :data     {:values accuracy-values}
                          :width    260
                          :mark     {:type "line", :legend nil}
                          :orient   "vertical"
                          :encoding {:x     {:field :batch, :type "ordinal"}
                                     :y     {:field :accuracy, :type "quantitative"}
                                     :color {:field :set, :type "nominal" "legend" nil}}})
                (clj->js {:width   360
                          :actions false}))

  (when-let [last-accuracy (:accuracy (last accuracy-values))]
    (-> (js/document.getElementById "accuracy-label")
        .-innerText
        (set! (gstring/format "last accuracy: %0.3f%" last-accuracy)))))

(defn- draw [data canvas]
  (let [[width height] [28 28]

        ctx        (.getContext canvas "2d")
        image-data (js/ImageData. width height)]

    (aset canvas "width" width)
    (aset canvas "height" height)

    (doseq [[i d] (mapv vector (range) data)
            :let [j (* i 4)
                  v (* d 255)]]
      (aset image-data "data" (+ j 0) v)
      (aset image-data "data" (+ j 1) v)
      (aset image-data "data" (+ j 2) v)
      (aset image-data "data" (+ j 3) 255))

    (.putImageData ctx image-data 0 0)))

(defn show-test-results [data label prediction]
  (let [div (.createElement js/document "div")]
    (aset div "className" "pred-container")

    (let [pred    (.createElement js/document "div")
          correct (if (= label prediction) "pred-correct" "pred-incorrect")]
      (aset pred "className" (str "pred " correct))
      (aset pred "innerText" (str "pred: " prediction))
      (.appendChild div pred))

    (let [canvas (.createElement js/document "canvas")]
      (aset canvas "className" "prediction-canvas")
      (draw data canvas)
      (.appendChild div canvas))

    (-> (js/document.getElementById "images")
        (.appendChild div))))

(def losses (atom []))
(def accuracies (atom []))

(defn update-loss [v]
  (let [batch (aget v "batch")
        loss  (aget v "loss")]
    (swap! losses conj {:batch batch :loss loss :set "train"})
    (plot-loss @losses)))

(defn update-accuracy [v]
  (let [batch (aget v "batch")
        acc   (aget v "acc")]
    (swap! accuracies conj {:batch batch :accuracy acc :set "train"})
    (plot-accuracy @accuracies)))

(defn reset-data! []
  (reset! losses [])
  (plot-loss [])

  (reset! accuracies [])
  (plot-accuracy @accuracies))

(defn reset-predictions []
  (-> (js/document.getElementById "images")
      .-innerHTML
      (set! "")))
