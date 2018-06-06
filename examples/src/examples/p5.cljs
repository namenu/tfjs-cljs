(ns examples.p5
  (:require [cljsjs.p5]))

;; Quil like interface of p5.js

(defn map-range [val low1 high1 low2 high2]
  (js/map val low1 high1 low2 high2))

(defn random [& args]
  (apply js/random args))


(defn sketch [state & opts]
  (let [{setup-fn         :setup
         update-fn        :update
         draw-fn          :draw
         mouse-pressed-fn :mouse-pressed,
         :or {mouse-pressed-fn identity}} opts]
    (doto js/window
      (aset "setup" (fn []
                      (swap! state setup-fn)))
      (aset "draw" (fn []
                     (swap! state update-fn)
                     (draw-fn @state)))
      (aset "mousePressed" (fn []
                             (swap! state mouse-pressed-fn))))))
