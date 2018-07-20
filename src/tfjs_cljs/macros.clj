(ns tfjs-cljs.macros)

(defn ->camelCase [s]
  (let [s (clojure.string/split s #"-")]
    (transduce (map clojure.string/capitalize) str (first s) (next s))))

(defmacro deftf [name]
  (let [params (gensym)
        cname  (symbol "js" (->camelCase (str "tf." name)))]
    `(defn ~name [& ~params]
       (apply ~cname ~params))))

(defmacro defconst [n keywords]
  (let [const-map (mapv (comp ->camelCase name) keywords)]
    `(def ~n (zipmap ~keywords ~const-map))))
