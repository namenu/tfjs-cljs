(ns tfjs-cljs.core)

(defmacro defvar
  [name value]
  `(def ~name (.variable js/tf ~value)))

(defmacro with-tidy
  [& body]
  `(.tidy js/tf (fn [] ~@body)))
