(ns tfjs-cljs.macros)

(defmacro deftf [name]
  (let [params (gensym)]
    `(defn ~name [~params]
       (. js/tf ~name ~params))))
