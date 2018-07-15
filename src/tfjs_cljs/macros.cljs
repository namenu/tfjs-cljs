(ns tfjs-cljs.macros)

(defn ->camelCase [s]
  (let [s (clojure.string/split s #"-")]
    (transduce (map clojure.string/capitalize) str (first s) (next s))))
