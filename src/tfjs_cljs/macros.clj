(ns tfjs-cljs.macros)

(defmacro deftf [name]
  (let [params (gensym)]
    `(defn ~name [~params]
       (. js/tf ~name ~params))))

(defmacro defconst
  [name consts]
  `(def ~name
     (zipmap ~consts
             (map (comp tfjs-cljs.macros/->camelCase name) ~consts))))
