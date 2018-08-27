; Color Classifier from "The Coding Train Session 7"
; Video: https://www.youtube.com/watch?v=y59-frfKR58&list=PLRqwX-V7Uu6bmMRCIoTi72aNWHo7epX4L&index=1

(ns examples.color-classifier
  (:require [cljsjs.p5]
            [reagent.core :as r]
            [tfjs-cljs.core :as tf :refer-macros [with-tidy with-fit-history]]
            [tfjs-cljs.core.train :as train]
            [tfjs-cljs.layers.models :as models]
            [tfjs-cljs.layers :as layers]))

(def config {:apiKey            "AIzaSyDPekCKX4ee6h9NVR2lEITGAM0XIHn-c7c",
             :authDomain        "color-classification.firebaseapp.com",
             :databaseURL       "https://color-classification.firebaseio.com",
             :projectId         "color-classification",
             :storageBucket     "",
             :messagingSenderId "590040209608"})
(defonce database (r/atom nil))
(defonce data (r/atom nil))
(defonce state (atom {}))

(def colors (map #(str % "-ish") ["red" "green" "blue" "orange" "yellow" "pink" "purple" "brown" "grey"]))
(def label-list (into {} (map-indexed #(vector %2 %1)) colors))

(def model
  (-> (models/sequential)

      ; hidden
      (models/add (layers/dense {:units      16
                                 :activation "sigmoid"
                                 :inputDim   3}))

      ; output
      (models/add (layers/dense {:units      9
                                 :activation "softmax"}))

      (models/compile {:optimizer (train/sgd 0.2)
                       :loss      "categoricalCrossentropy"})))

(def loss (r/atom nil))
(def prediction (r/atom ""))

(def test-color (r/atom {:r 255 :g 0 :b 0}))

(defn slider [param value]
  [:input {:type      "range" :value value :min 0 :max 255
           :style     {:width "200px"}
           :on-change (fn [e]
                        (swap! test-color assoc param (.. e -target -value)))}])
(defn record->rgb [r]
  (let [rgb ((juxt :r :g :b) r)]
    (map #(/ % 255) rgb)))

(defn got-data [results]
  (reset! data (js->clj (.val results) :keywordize-keys true)))

(defn setup-firebase []
  (when-not @database
    (js/firebase.initializeApp (clj->js config))
    (reset! database (js/firebase.database)))

  (when-not @data
    (let [ref (.ref @database "colors")
          ref (.limitToLast ref 5000)]
      (.once ref "value" got-data))))

(defn train [model xs ys]
  (let [callbacks {:onEpochEnd (fn [num logs]
                                 (reset! loss logs.loss)
                                 (console.log "Epoch: " num logs.val_loss))
                   :onBatchEnd tf/next-frame}

        options   {:epochs          10
                   :validationSplit 0.1
                   :shuffle         true
                   :callbacks       callbacks}]
    (models/fit model xs ys options)))

(defn do-train []
  (when-let [data (vals @data)]
    (let [colors (map record->rgb data)
          labels (into []
                       (comp
                         (map :label)
                         (map label-list))
                       data)

          xs     (tf/tensor2d colors)
          ys     (tf/one-hot labels 9)
          ]

      (train model xs ys)

      #_#_(tf/dispose xs)
          (tf/dispose ys))))

(defn app []
  [:div

   [:div {:id "test-color"}]

   (let [{:keys [r g b]} @test-color]
     [:div
      [:div
       "R"
       [slider :r r]]
      [:div
       "G"
       [slider :g g]]
      [:div
       "B"
       [slider :b b]]])

   [:button {:on-click #(do-train)} "Train!"]
   [:div "Prediction:" @prediction]
   [:div "Loss: " @loss]])

(defn setup [sketch]
  (.createCanvas sketch 200 200))

(defn draw [sketch]
  (let [{:keys [r g b]} @test-color]
    (.background sketch r g b))

  (with-tidy [xs (tf/tensor2d [(record->rgb @test-color)])
              results (models/predict model xs)
              index (-> results
                        (tf/arg-max 1)
                        (tf/data-sync)
                        (first))]
    (reset! prediction (nth colors index))))

(defn ^export run []
  (r/render [app]
            (js/document.getElementById "app"))

  (setup-firebase)

  (let [s (fn [sketch]
            (aset sketch "setup" #(setup sketch))
            (aset sketch "draw" #(draw sketch)))]
    (js/p5. s "test-color")))

(run)
