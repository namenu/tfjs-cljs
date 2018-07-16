(ns tfjs-cljs.macros)

(defn ->camelCase [s]
  (let [s (clojure.string/split s #"-")]
    (transduce (map clojure.string/capitalize) str (first s) (next s))))

(defmacro deftf [name]
  (let [params (gensym)
        cname  (symbol (->camelCase (str name)))]
    `(defn ~name [~params]
       (. js/tf ~cname ~params))))

(defmacro defconst [n keywords]
  (let [const-map (mapv (comp ->camelCase name) keywords)]
    `(def ~n (zipmap ~keywords ~const-map))))
